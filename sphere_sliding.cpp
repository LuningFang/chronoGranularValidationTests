// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2020 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Luning Fang
// =============================================================================
// validation test: single sphere rolling on a plane modeled as a planeBC
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>
#include "ChGranularDemoUtils.hpp"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"
#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::granular;


void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file> \n " << std::endl;
}

int main(int argc, char* argv[]) {
    sim_param_holder params;

    // Some of the default values are overwritten by user via command line
    if (argc != 2 || ParseJSON(argv[1], params) == false) {
        ShowUsage(argv[0]);
        return 1;
    }

    // big domain dimension
    params.box_X = 100;
    params.box_Y = 100;
    params.box_Z = 8 * params.sphere_radius;

    // rolling friction coefficient
    float rolling_fric_coeff = 0.1f;
    params.rolling_friction_coeffS2S = rolling_fric_coeff;
    params.rolling_friction_coeffS2W = rolling_fric_coeff;
    
  // Setup simulation
    ChSystemGranularSMC gran_sys(params.sphere_radius, params.sphere_density,
                                 make_float3(params.box_X, params.box_Y, params.box_Z));

    ChGranularSMC_API apiSMC;
    apiSMC.setGranSystem(&gran_sys);

    // create plane BC at the bottom of BD
    float plane_pos[3] = {0.0f, 0.0f, -3*params.sphere_radius};
    float plane_normal[3] = {0.0f, 0.0f, 1.0f};
    size_t plane_bc_id = gran_sys.Create_BC_Plane(plane_pos, plane_normal, true);

    // calcuate settled position to make sure sphere barely touching the planeBC
	double mass = 4.0/3.0 * CH_C_PI * pow(params.sphere_radius,3) * params.sphere_density;
	double penetration = pow(mass * abs(params.grav_Z) / params.normalStiffS2S * std::sqrt(params.sphere_radius), 2.0/3.0);
	double settled_pos = -3*params.sphere_radius + params.sphere_radius - penetration;

    // assign initial condition for the sphere
    float initialVelo = 0.5f;
    std::vector<ChVector<float>> body_point;
	body_point.push_back(ChVector<float>(-3.0f, -4.0f, settled_pos));
	std::vector<ChVector<float>> velocity;
	velocity.push_back(ChVector<float>(initialVelo, 0.0f, 0.0f));
    apiSMC.setElemsPositions(body_point, velocity);

	gran_sys.setPsiFactors(params.psi_T, params.psi_L);

	// set normal force model
    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);

	// set tangential force model
	gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);		
	gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
	gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
	gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
	gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
	gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);


    // set cohesion and adhesion model
	gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);

    // set gravity
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

   
	gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CHUNG);

	// set rolling friction model
	gran_sys.set_rolling_mode(GRAN_ROLLING_MODE::SCHWARTZ);
	gran_sys.set_rolling_coeff_SPH2SPH(params.rolling_friction_coeffS2S);
	gran_sys.set_rolling_coeff_SPH2WALL(params.rolling_friction_coeffS2W);

    // set time integrator
    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);


    gran_sys.set_BD_Fixed(true);

    gran_sys.setVerbose(params.verbose);
    gran_sys.initialize();
  

    float curr_time = 0;
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> omega;
    float reaction_force[3] = {1E6, 1E6, 1E6};

    float ratio = 1E7;

    float weight = mass * std::abs(params.grav_Z);
    float slidingFr = weight * params.static_friction_coeffS2W;

    printf("mass = %f, normal force = %f, sliding friction force = %f\n", mass, weight, slidingFr);
    
    float Fr_threshold = 1E-6;
    float Fr_prev = reaction_force[0];
    float Fr_diff = 1E3;

	printf("time, ball_px, ball_pz, vel_x, vel_z, omega.y\n");
//	while (curr_time < params.time_end && Fr_diff > Fr_threshold) {
	while (curr_time < params.time_end) {

        gran_sys.advance_simulation(params.step_size);
        curr_time += params.step_size;

        pos   = apiSMC.getPosition(0);
        velo  = apiSMC.getVelo(0);
        omega = apiSMC.getAngularVelo(0);
        Fr_prev = reaction_force[0];
        gran_sys.getBCReactionForces(plane_bc_id, reaction_force);
        Fr_diff = std::abs(Fr_prev - reaction_force[0]);

        if (std::abs(omega.y()) > 1E-8)
            ratio = std::abs(velo.x()/omega.y());

        // printf("%e, %e, %e, %e, %e, %e, %e, %e, %e\n", curr_time, pos.x(), pos.z(), velo.x(), velo.z(), omega.y(), reaction_force[0], reaction_force[1], reaction_force[2]);

		}

    // printf("mass = %f, normal force = %f, sliding friction force = %f\n", mass, weight, slidingFr);

    return 0;
}
