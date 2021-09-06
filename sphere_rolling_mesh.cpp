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
// Rolling ball on the ground, modeled as mesh, z pointing down
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>

#include "chrono/core/ChGlobal.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/assets/ChTriangleMeshShape.h"

#include "chrono_gpu/ChGpuData.h"
#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_gpu/utils/ChGpuJsonParser.h"
#include "chrono_gpu/ChGpuData.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::gpu;

int main(int argc, char* argv[]) {
    double mu_k = 0.3;
    double cor = 0.0;
    double step_size = 3e-5;
    double sphere_mass = 1.0;    // 1kg
    double sphere_radius = 0.5;  // diameter 1m
    double sphere_volume = 4. / 3. * CH_C_PI * std::pow(sphere_radius, 3);
    double sphere_density = sphere_mass / sphere_volume;
    // double time_end = 2;
    // double time_settle = 1.0f;
    // double time_roll = 1.0f;
    double sphere_inertia = 0.4 * sphere_mass * sphere_radius * sphere_radius;

    double time_settle = 0.3f;
    double time_roll = 0.5f;
    double time_end = time_settle + time_roll;

    float mu_roll = 0.0f;
    double rolling_friction_coeffS2S = mu_roll;
    double rolling_friction_coeffS2W = mu_roll;
    double rolling_friction_coeffS2M = mu_roll;

    float v_init_mag = 1;  // initial velo 1m/s

    double box_X = 10 * sphere_radius;
    double box_Y = 10 * sphere_radius;
    double box_Z = 10 * sphere_radius;

    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -9.81;

    bool use_material_based_model = true;
    // Setup simulation
    ChSystemGpuMesh gran_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    gran_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gran_sys.SetRollingCoeff_SPH2SPH(rolling_friction_coeffS2S);
    gran_sys.SetRollingCoeff_SPH2WALL(rolling_friction_coeffS2W);
    gran_sys.SetRollingCoeff_SPH2MESH(rolling_friction_coeffS2M);

    gran_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
    gran_sys.SetStaticFrictionCoeff_SPH2SPH(mu_k);
    gran_sys.SetStaticFrictionCoeff_SPH2WALL(mu_k);
    gran_sys.SetStaticFrictionCoeff_SPH2MESH(mu_k);

    switch (use_material_based_model) {
        case true: {
            double youngs_modulus = 2.0e6;  //
            double poisson_ratio = 0.3;

            gran_sys.UseMaterialBasedModel(true);

            gran_sys.SetYoungModulus_SPH(youngs_modulus);
            gran_sys.SetYoungModulus_WALL(youngs_modulus);
            gran_sys.SetYoungModulus_MESH(youngs_modulus);

            gran_sys.SetRestitution_SPH(cor);
            gran_sys.SetRestitution_WALL(cor);
            gran_sys.SetRestitution_MESH(cor);

            gran_sys.SetPoissonRatio_SPH(poisson_ratio);
            gran_sys.SetPoissonRatio_WALL(poisson_ratio);
            gran_sys.SetPoissonRatio_MESH(poisson_ratio);

            break;
        }
        case false: {
            double kn = 1e7;
            double gn = 2e4;
            double kt = 2e6;
            double gt = 5000;

            gran_sys.UseMaterialBasedModel(false);
            gran_sys.SetKn_SPH2SPH(kn);
            gran_sys.SetKn_SPH2WALL(kn);
            gran_sys.SetKn_SPH2MESH(kn);

            gran_sys.SetGn_SPH2SPH(gn);
            gran_sys.SetGn_SPH2WALL(gn);
            gran_sys.SetGn_SPH2MESH(gn);

            gran_sys.SetKt_SPH2SPH(kt);
            gran_sys.SetKt_SPH2WALL(kt);
            gran_sys.SetKt_SPH2MESH(kt);

            gran_sys.SetGt_SPH2SPH(gt);
            gran_sys.SetGt_SPH2WALL(gt);
            gran_sys.SetGt_SPH2MESH(gt);
        }
    }

    // sphere initial position and velocity
    ChVector<float> initial_position(-.2f, 0.0f, sphere_radius + 0.01);
    //    ChVector<float> initial_position(0.0f, 0.0f, sphere_radius + 0.01);
    ChVector<float> initial_velo(0.0, 0.0, -0.1f);

    gran_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CENTERED_DIFFERENCE);

    std::vector<ChVector<float>> body_points;
    body_points.push_back(initial_position);

    std::vector<ChVector<float>> body_vels(1, initial_velo);
    gran_sys.SetParticles(body_points, body_vels);

    gran_sys.SetFixedStepSize(step_size);

    // Add the plate mesh to the GPU system
    float scale_xy = 1.0f;
    float scale_z = 1.0f;
    ChVector<> scaling(scale_xy, scale_xy, scale_z);
    ChVector<> plate_position(0, 0, -0.05);
    float plate_mass = 1000;
    auto plate_mesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    plate_mesh->LoadWavefrontMesh(GetChronoDataFile("models/plate.obj"), true, false);
    plate_mesh->Transform(plate_position, ChMatrix33<>(scaling));
    auto plate_mesh_id = gran_sys.AddMesh(plate_mesh, plate_mass);

    gran_sys.SetBDFixed(true);
    // gran_sys.SetVerbosity(CHGPU_VERBOSITY::METRICS);
    gran_sys.EnableMeshCollision(true);
    gran_sys.Initialize();

    float curr_time = 0;
    int fps = 1000;
    int currframe = 0;
    // string output_dir = "rolling";
    // filesystem::create_directory(filesystem::path(output_dir));
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> angular_velo;

    double KE;

    while (curr_time < time_settle) {
        gran_sys.AdvanceSimulation(step_size);
        curr_time += step_size;
        currframe++;

        pos = gran_sys.GetParticlePosition(0);
        velo = gran_sys.GetParticleVelocity(0);

        KE = 0.5 * sphere_mass * velo.Length2();

        if (KE < 1.0e-15) {
            std::cout << "[settling] KE falls below threshold after " << curr_time << " s \n";
            printf("t,%e,vx,%e,vy,%e,vz,%e,wx,%e,wy,%e,wz,%e\n", curr_time, velo.x(), velo.y(), velo.z(),
                   angular_velo.x(), angular_velo.y(), angular_velo.z());

            break;
        }

        if (currframe % fps == 0) {
            // printf("t,%e,vx,%e,vy,%e,vz,%e,wx,%e,wy,%e,wz,%e\n", curr_time, velo.x(), velo.y(), velo.z(),
            // angular_velo.x(), angular_velo.y(), angular_velo.z());
        }
    }

    double start_time = curr_time;
    double endTime = start_time + time_roll;
    ChVector<double> roll_velo(1.0f, 0.0f, 0.0f);
    gran_sys.SetParticleVelocity(0, roll_velo);

    while (curr_time < endTime) {
        gran_sys.AdvanceSimulation(step_size);
        curr_time += step_size;
        currframe++;
        pos = gran_sys.GetParticlePosition(0);
        velo = gran_sys.GetParticleVelocity(0);
        angular_velo = gran_sys.GetParticleAngVelocity(0);

        KE = 0.5 * sphere_mass * velo.Length2() + 0.5 * sphere_inertia * angular_velo.Length2();

        if (KE < 1.0e-9) {
            std::cout << "[rolling] KE falls below threshold after " << curr_time - start_time << " s \n";
            break;
        }

        // if (currframe%100 == 0){
        //     // std::cout << "t: "  << curr_time << ", velo: " << velo.x() << ", omic: " << angular_velo.y() <<
        //     std::endl; printf("time = %e, %e, %e\n", curr_time - start_time, velo.x(), angular_velo.y());

        // }
    }

    std::cout << "ending angular velocity: " << angular_velo.x() << ", " << angular_velo.y() << ", " << angular_velo.z()
              << std::endl;

    return 0;
}
