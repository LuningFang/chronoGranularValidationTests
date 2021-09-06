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
// validation test: single sphere dropped on the ground to test boucing
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>

#include "GpuDemoUtils.hpp"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_gpu/physics/ChSystemGpu.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::gpu;

int main(int argc, char* argv[]) {
    double sphere_radius = 0.5;
    double sphere_density = 7.4;
    double youngs_modulus = 5e6;
    double poisson_ratio = 0.3;
    double mu_s = 0;
    double cor = 0.4;
    double step_size = 1e-5;
    double time_end = 0.26;

    // big domain dimension
    double box_X = 8 * sphere_radius;
    double box_Y = 8 * sphere_radius;
    double box_Z = 8 * sphere_radius;

    // Setup simulation
    ChSystemGpu gpu_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    // create plane BC at the bottom of BD
    ChVector<float> plane_pos(0.0f, 0.0f, (float)(-3.0f * sphere_radius));
    ChVector<float> plane_normal(0, 0, 1);
    size_t plane_bc_id = gpu_sys.CreateBCPlane(plane_pos, plane_normal, true);

    double mass = 4.0 / 3.0 * CH_C_PI * pow(sphere_radius, 3) * sphere_density;

    // assign initial condition for the sphere
    std::vector<ChVector<float>> body_point;
    body_point.push_back(ChVector<float>(-0.0f, -0.0f, -sphere_radius));
    gpu_sys.SetParticles(body_point);

    float psi_T = 32.0f;
    float psi_L = 256.0f;
    gpu_sys.SetPsiFactors(psi_T, psi_L);

    // set normal force model
    gpu_sys.SetYoungModulus_SPH(youngs_modulus);
    gpu_sys.SetYoungModulus_WALL(youngs_modulus);

    gpu_sys.SetRestitution_SPH(cor);
    gpu_sys.SetRestitution_WALL(cor);
    gpu_sys.SetPoissonRatio_SPH(poisson_ratio);
    gpu_sys.SetPoissonRatio_WALL(poisson_ratio);

    gpu_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
    gpu_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s);
    gpu_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s);

    // set gravity
    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -980.0f;

    gpu_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    // set time integrator
    gpu_sys.SetFixedStepSize(step_size);
    gpu_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::FORWARD_EULER);

    gpu_sys.SetVerbosity(CHGPU_VERBOSITY::QUIET);

    gpu_sys.SetBDFixed(true);
    gpu_sys.Initialize();

    float curr_time = 0;
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> omega;

    ChVector<float> reaction_force;

    printf("time, ball_pz, vel_z, omega.y\n");
    while (curr_time < time_end) {
        gpu_sys.AdvanceSimulation(step_size);
        curr_time += step_size;

        pos = gpu_sys.GetParticlePosition(0);
        velo = gpu_sys.GetParticleVelocity(0);

        gpu_sys.GetBCReactionForces(plane_bc_id, reaction_force);

        printf("%f, %e, %e, %e\n", curr_time, pos.z(), velo.z(), reaction_force.z());
    }

    return 0;
}
