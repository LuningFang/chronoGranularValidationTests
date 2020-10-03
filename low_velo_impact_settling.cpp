// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Luning Fang
// =============================================================================
// Granular material settling in a square container to generate the bed for ball
// drop test, once settled positions are written
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include "chrono/core/ChGlobal.h"
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChForce.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/timestepper/ChTimestepper.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/physics/ChGranularTriMesh.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"
#include "ChGranularDemoUtils.hpp"

using namespace chrono;
using namespace chrono::granular;

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file>" << std::endl;
    std::cout << "OR " + name + " <json_file> "  + " <input_file> " <<std::endl;
}

float getMass(sim_param_holder& params){
    float rad = params.sphere_radius;
    float density = params.sphere_density;

    float volume = 4.0f/3.0f * CH_C_PI * std::pow(rad, 3);
    float mass = volume * density;
    return mass;
}

// calculate kinetic energy of the system
float getSystemKE(sim_param_holder &params, ChGranularSMC_API &apiSMC, int numSpheres){
    float sysKE = 0.0f;
    float sphere_KE;
    ChVector<float> angularVelo;
    ChVector<float> velo;
    float mass = getMass(params);
    float inertia = 0.4f * mass * std::pow(params.sphere_radius,2);

    for (int i = 0; i < numSpheres; i++){
        angularVelo = apiSMC.getAngularVelo(i);
        velo = apiSMC.getVelo(i);
        sphere_KE = 0.5f * mass * velo.Length2() + 0.5f * inertia * angularVelo.Length2();
        sysKE = sysKE + sphere_KE;
    }
    return sysKE;
}

int main(int argc, char* argv[]) {
    sim_param_holder params;
    bool useCheckpointInput = false;

    if ((argc != 2 && argc != 3) || ParseJSON(argv[1], params) == false){
        ShowUsage(argv[0]);
        return 1;
    }
    if (argc == 3 ) {
        useCheckpointInput = true;
    }

    float iteration_step = params.step_size;

    ChSystemGranularSMC gran_sys(params.sphere_radius, params.sphere_density,
                                 make_float3(params.box_X, params.box_Y, params.box_Z));

    ChGranularSMC_API apiSMC;
    apiSMC.setGranSystem(&gran_sys);

    // set cylinder boundary condition
    float cyl_center[3] = {0.0f, 0.0f, 0.0f};
    float cyl_rad = std::min(params.box_X, params.box_Y)/2.0f;
    size_t cyl_id = gran_sys.Create_BC_Cyl_Z(cyl_center, cyl_rad, false, true);
    

    // declare particle positon vector
    std::vector<chrono::ChVector<float>> body_points; 
    if (useCheckpointInput == true){
        body_points = loadPositionCheckpoint<float>(argv[2]);
		std::cout << "reading position input success from " << argv[2]<<std::endl;
	}
    else
    {
        utils::HCPSampler<float> sampler(2.02 * params.sphere_radius); 
        ChVector<float> center(0.0f, 0.0f, 0.0f);
        body_points = sampler.SampleCylinderZ(center, cyl_rad-params.sphere_radius, params.box_Z/2 - params.sphere_radius);
    }
    
    int numSpheres = body_points.size();
    std::cout << "numbers of particles created: " << numSpheres << std::endl;
    apiSMC.setElemsPositions(body_points);

    gran_sys.set_BD_Fixed(true);

    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);

    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);

    gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
    gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);

    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);

    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
    gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
    gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);

    gran_sys.setOutputMode(params.write_mode);
    gran_sys.setVerbose(params.verbose);
    filesystem::create_directory(filesystem::path(params.output_dir));

    gran_sys.initialize();

    unsigned int out_fps = 5;
    std::cout << "Rendering at " << out_fps << "FPS" << std::endl;

    unsigned int out_steps = (unsigned int)(1.0 / (out_fps * iteration_step));

    int currframe = 0;
    unsigned int curr_step = 0;

    clock_t start = std::clock();
    for (double t = 0; t < (double)params.time_end; t += iteration_step, curr_step++) {
        gran_sys.advance_simulation(iteration_step);
        float KE;
        if (curr_step % out_steps == 0) {
            KE = getSystemKE(params, apiSMC, numSpheres);
            std::cout << ", time = " << t << ", KE = " << KE * 1E-5 << ", max z: " << gran_sys.get_max_z() << std::endl;

		}
    }

    char filename[100];
    sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe++);
    gran_sys.writeFile(std::string(filename));


    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;

    // delete[] meshPosRot;
    // delete[] meshVel;

    return 0;
}
