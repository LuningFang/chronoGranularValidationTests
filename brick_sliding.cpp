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
// validation test: single sphere sliding test (angular acceleration need to
// be turned off inside the solver since granular models particles only)
// ** this could be hard to become a unit test :(((
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>
#include <chrono>
#include "ChGranularDemoUtils.hpp"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono_granular/utils/ChCudaMathUtils.cuh"

using namespace chrono;
using namespace chrono::granular;

enum integrator { CENTERED_DIFFERENCE = 0, EXTENDED_TAYLOR = 1, FORWARD_EULER = 2, CHUNG = 3 };

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("need input for mu_k and integrator\n");
        return -1;
    }

    double mu_k = atof(argv[1]);

    integrator mySolver = (integrator)(std::atoi(argv[2]));

    double sphere_radius = 0.5;
    double sphere_volume = 4. / 3. * CH_C_PI * std::pow(sphere_radius, 3);
    double sphere_mass = 1.0;
    double sphere_density = sphere_mass / sphere_volume;
    double youngs_modulus = 2.0e5;
    double cor = 0.0;
    double poisson_ratio = 0.3;
    double step_size = 2.0e-5;
    double time_end = 15;

    // big domain dimension
    double box_X = 30 * sphere_radius;
    double box_Y = 30 * sphere_radius;
    double box_Z = 30 * sphere_radius;

    // Setup simulation
    ChSystemGranularSMC gran_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    ChGranularSMC_API apiSMC;
    apiSMC.setGranSystem(&gran_sys);

    // create plane BC at the bottom of BD
    float plane_pos[3] = {0.0f, 0.0f, (float)(-3.0 * sphere_radius)};
    float plane_normal[3] = {0.0f, 0.0f, 1.0f};
    size_t plane_bc_id = gran_sys.Create_BC_Plane(plane_pos, plane_normal, true);

    // assign initial condition for the sphere
    std::vector<ChVector<float>> body_point;
    double init_pos = -box_X / 2. + 2. * sphere_radius;
    body_point.push_back(ChVector<float>(init_pos, -0.0f, -2.0 * sphere_radius));
    apiSMC.setElemsPositions(body_point);

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

    // set gravity
    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -9.81f;
    double3 grav_vec{grav_X, grav_Y, grav_Z};

    gran_sys.set_gravitational_acceleration(grav_X, grav_Y, grav_Z);

    // set time integrator
    gran_sys.set_fixed_stepSize(step_size);
    switch (mySolver) {
        case integrator::CENTERED_DIFFERENCE:
            gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
            break;
        case integrator::FORWARD_EULER:
            gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::FORWARD_EULER);
            break;
        case integrator::EXTENDED_TAYLOR:
            gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::EXTENDED_TAYLOR);
            break;
        case integrator::CHUNG:
            gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CHUNG);
            break;
    }

    GRAN_VERBOSITY verbose = QUIET;
    gran_sys.setVerbose(verbose);
    gran_sys.set_BD_Fixed(true);
    gran_sys.initialize();

    float curr_time = 0;
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> omega;

    float reaction_force[3];

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // wait for the sphere to settle first
    while (curr_time < 0.08) {
        gran_sys.advance_simulation(step_size);
        curr_time += step_size;

        pos = apiSMC.getPosition(0);
        velo = apiSMC.getVelo(0);

        gran_sys.getBCReactionForces(plane_bc_id, reaction_force);

        // printf("%f, %e, %e, %e\n", curr_time, pos.z(), velo.z(), reaction_force[2]);
    }

    double3 init_v = make_double3(5.0, 0.0, 0.0);
    gran_sys.setVelocity(0, init_v);

    while (curr_time < 1.3) {
        gran_sys.advance_simulation(step_size);
        curr_time += step_size;

        pos = apiSMC.getPosition(0);
        velo = apiSMC.getVelo(0);
        omega = apiSMC.getAngularVelo(0);

        gran_sys.getBCReactionForces(plane_bc_id, reaction_force);

        if (std::abs(velo.x()) < 1e-7) {
            break;
        }

        // printf("%f, %e, %e, %e, %e\n", curr_time, pos.x(), velo.x(), omega.y(), reaction_force[0]);
        // printf("%f, %e, %e, %e, %e\n", curr_time, pos.x(), velo.x(), velo.y(), velo.z());
    }

    // analytical distance
    double d_ref = std::abs(Length2(init_v) / (2 * mu_k * Length(grav_vec)));
    double d_sim = pos.x() - init_pos;
    double err = (std::abs(d_ref - d_sim) / d_ref) * 100;

    // LULUTODO: ISSUE!!! LOOK INTO THIS velo in z direction in order of
    // 1e-4 while x direction can get to 1e-13
    printf("mu = %.2f, integrator: %d, time: %e, ending_velo: %e, ending dist, %e, analytical, %e, err: %.4f%%\n", mu_k,
           mySolver, curr_time, velo.Length(), d_sim, d_ref, err);

    // check actual travel distance against reference

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds" << std::endl;

    return 0;
}
