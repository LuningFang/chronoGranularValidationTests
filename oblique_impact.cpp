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
// validation test: oblique contact, single sphere test,
// can not be made into a unit test, hack needed to set gravitational force to zero
// in applyGravity function in ChGranularGPU_SMC. Do not set it at user-end. Gravity
// values are needed for calculating K_UU2SU.
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
#include "chrono_granular/utils/ChCudaMathUtils.cuh"

using namespace chrono;
using namespace chrono::granular;

// enum integrator {CENTERED_DIFFERENCE = 0, EXTENDED_TAYLOR = 1, FORWARD_EULER = 2, CHUNG = 3};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("need input for impact angle (deg) and cor\n");
        return -1;
    }

    double mu_k = 0.3;
    double theta_deg = atof(argv[1]);
    double theta_rad = theta_deg / 180. * CH_C_PI;
    double cor = atof(argv[2]);

    double sphere_radius = 0.5;
    double sphere_volume = 4. / 3. * CH_C_PI * std::pow(sphere_radius, 3);
    double sphere_mass = 1.0;
    double sphere_density = sphere_mass / sphere_volume;
    double youngs_modulus = 2.0e5;
    double poisson_ratio = 0.3;
    double step_size = 2.0e-5;

    double time_end = 3.;

    // big domain dimension
    double box_X = 100 * sphere_radius;
    double box_Y = 100 * sphere_radius;
    double box_Z = 30 * sphere_radius;

    // Setup simulation
    ChSystemGranularSMC gran_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    ChGranularSMC_API apiSMC;
    apiSMC.setGranSystem(&gran_sys);

    // set gravity
    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -9.81f;
    double3 grav_vec{grav_X, grav_Y, grav_Z};
    gran_sys.set_gravitational_acceleration(grav_X, grav_Y, grav_Z);

    // create plane BC at the bottom of BD
    float plate_pos_z = -3.0 * sphere_radius;
    float plane_pos[3] = {0.0f, 0.0f, plate_pos_z};
    float plane_normal[3] = {0.0f, 0.0f, 1.0f};
    size_t plane_bc_id = gran_sys.Create_BC_Plane(plane_pos, plane_normal, true);

    // assign initial condition for the sphere
    std::vector<ChVector<float>> body_point;
    double drop_dist = sphere_radius;

    double vn_contact_theoretical = std::sqrt(2 * std::abs(grav_Z) * drop_dist);
    double v_in = vn_contact_theoretical / std::tan(theta_rad);

    body_point.push_back(
        ChVector<float>(-box_X / 2. + 2. * sphere_radius, -0.0f, plate_pos_z + drop_dist + sphere_radius));
    std::vector<ChVector<float>> init_velo;
    // init_velo.push_back(ChVector<float>(v_in, 0.0, 0.0));
    init_velo.push_back(ChVector<float>(v_in, 0.0, -vn_contact_theoretical));

    apiSMC.setElemsPositions(body_point, init_velo);

    float psi_T = 32.0f;
    float psi_L = 256.0f;
    gran_sys.setPsiFactors(psi_T, psi_L);

    // set normal force model
    gran_sys.set_YoungsModulus_SPH(youngs_modulus);
    gran_sys.set_YoungsModulus_WALL(youngs_modulus);
    gran_sys.set_COR_SPH(cor);
    gran_sys.set_COR_WALL(cor);
    gran_sys.set_PoissonRatio_SPH(poisson_ratio);
    gran_sys.set_PoissonRatio_WALL(poisson_ratio);

    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_static_friction_coeff_SPH2SPH(mu_k);
    gran_sys.set_static_friction_coeff_SPH2WALL(mu_k);

    // set cohesion and adhesion model
    // gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    // gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);

    // set time integrator
    gran_sys.set_fixed_stepSize(step_size);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CHUNG);

    GRAN_VERBOSITY verbose = QUIET;
    gran_sys.setVerbose(verbose);
    gran_sys.set_BD_Fixed(true);
    gran_sys.initialize();

    float curr_time = 0;
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> omega;

    float reaction_force[3];
    double tan_theta;
    double vn_contact;

    // wait for the sphere to settle first
    while (curr_time < time_end) {
        gran_sys.advance_simulation(step_size);
        curr_time += step_size;

        pos = apiSMC.getPosition(0);
        velo = apiSMC.getVelo(0);
        omega = apiSMC.getAngularVelo(0);
        gran_sys.getBCReactionForces(plane_bc_id, reaction_force);

        if (std::abs(reaction_force[0]) > 1e-7 && std::abs(reaction_force[2]) > 1e-7) {
            tan_theta = std::abs(velo.x() / velo.z());
            vn_contact = std::abs(velo.z());
            break;
        }

        // printf("time, %f, vx, %e, vz, %e, omic, %e\n", curr_time, velo.x(), velo.z(), omega.y());
    }

    while (curr_time < time_end) {
        gran_sys.advance_simulation(step_size);
        curr_time += step_size;

        pos = apiSMC.getPosition(0);
        velo = apiSMC.getVelo(0);
        omega = apiSMC.getAngularVelo(0);
        gran_sys.getBCReactionForces(plane_bc_id, reaction_force);
        // sphere bouces up and no contact between the sphere and wall
        if (velo.z() > 0 && std::abs(reaction_force[0]) < 1e-7 && std::abs(reaction_force[2]) < 1e-7) {
            break;
        }

        // printf("time, %f, vx, %e, vz, %e, omic, %e\n", curr_time, velo.x(), velo.z(), omega.y());
    }

    double cor_t_analytical = 1. - mu_k * (1 + cor) / tan_theta;
    double vt_end = cor_t_analytical * v_in;
    double omega_analytical = 2.5 * mu_k * (1 + cor) * vn_contact / sphere_radius;
    printf("cor, %.2f, theta, %f, vx_init, %f, vx_end, %e, vx_th, %e, omega, %e, omega_th, %e\n", cor, theta_deg, v_in,
           velo.x(), vt_end, omega.y(), omega_analytical);

    return 0;
}
