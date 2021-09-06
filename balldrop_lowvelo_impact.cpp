// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2021 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Luning Fang
// =============================================================================
// Low-velocity impact test of a sphere (mesh) dropped on the a bed of settle
// granular material
// =============================================================================

#include <iostream>
#include <vector>
#include <string>

#include "chrono/core/ChGlobal.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/assets/ChSphereShape.h"

#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_gpu/utils/ChGpuJsonParser.h"
#include "chrono_gpu/ChGpuData.h"
#include "GpuDemoUtils.hpp"
#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::gpu;

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <input_file> " + " <drop height, 1 ~ 7 cm> " << std::endl;
}

// calculate kinetic energy of the system
float getSystemKE(double sphere_denisty, double sphere_radius, ChSystemGpuMesh& gpu_sys, int nb) {
    float sysKE = 0.0f;
    float sphere_KE;
    ChVector<float> angularVelo;
    ChVector<float> velo;

    float volume = 4.0f / 3.0f * CH_C_PI * std::pow(sphere_radius, 3);
    float mass = volume * sphere_denisty;

    float inertia = 0.4f * mass * std::pow(sphere_radius, 2);

    for (int i = 0; i < nb; i++) {
        angularVelo = gpu_sys.GetParticleAngVelocity(i);
        velo = gpu_sys.GetParticleVelocity(i);
        sphere_KE = 0.5f * mass * velo.Length2() + 0.5f * inertia * angularVelo.Length2();
        sysKE = sysKE + sphere_KE;
    }
    return sysKE;
}

int main(int argc, char* argv[]) {
    // unit gcm
    double sphere_radius = 0.5f;  // 1cm diameter
    double sphere_density = 2.5;
    double box_X = 31.5;
    double box_Y = 31.5;
    // double box_Z = 20.0;

    double box_Z = 50.0;  // increase box_Z dimension to include sphere mesh

    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -980.0f;

    float time_end = 0.3;
    if (argc != 2) {
        ShowUsage(argv[0]);
        return 1;
    }

    float step_size = 1e-6;

    ChSystemGpuMesh gran_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    // declare particle positon vector
    std::vector<chrono::ChVector<float>> body_points;
    body_points = loadPositionCheckpoint<float>(argv[1]);
    std::cout << "sucessfully load file from: " << argv[1] << std::endl;
    gran_sys.SetParticles(body_points);

    // find highest point within a certain radius
    // TODO: look at Radu's code
    double max_z = -1000;
    for (int i = 0; i < body_points.size(); i++) {
        double dist =
            std::sqrt(body_points.at(i).x() * body_points.at(i).x() + body_points.at(i).y() * body_points.at(i).y());

        if (dist < 6 * sphere_radius && body_points.at(i).z() > max_z) {
            max_z = body_points.at(i).z();
        }
    }

    // parameters for the impact test
    float initial_height = 4.9;
    std::cout << "drop height: " << initial_height << "cm. " << std::endl;

    // TODO: find a better way to describe the surface
    float initial_surface = max_z + sphere_radius;
    std::cout << "highest point at " << initial_surface << "cm" << std::endl;
    float initial_volume = box_X * box_Y * (initial_surface + box_Z / 2.0f);
    int numSpheres = body_points.size();
    std::cout << "initialize " << numSpheres << " particles" << std::endl;
    float volume_per_particle = 4.0 / 3.0 * CH_C_PI * std::pow(sphere_radius, 3);
    float mass_per_particle = volume_per_particle * sphere_density;

    float bulk_mass = numSpheres * mass_per_particle;
    // TODO: cout bulk density and packing frac during settling stage
    float bulk_density = bulk_mass / initial_volume;
    float packing_frac = numSpheres * volume_per_particle / initial_volume;
    std::cout << "bulk density is " << bulk_density << "g/cm3" << std::endl;
    std::cout << "packing frac is " << packing_frac << std::endl;

    // create cylinder containter
    ChVector<float> cyl_center(0.0f, 0.0f, 0.0f);
    float cyl_rad = std::min(box_X, box_Y) / 2.0f;
    size_t cyl_id = gran_sys.CreateBCCylinderZ(cyl_center, cyl_rad, false, true);

    // create bottom boundary at location -15
    double bottom_loc = -15.0f;

    // create bottom plane boundary condition with its position and normal
    ChVector<float> bottom_wall_pos(0.0f, 0.0f, bottom_loc);
    ChVector<float> bottom_wall_nrm(0.0f, 0.0f, 1.0f);
    size_t bottom_wall = gran_sys.CreateBCPlane(bottom_wall_pos, bottom_wall_nrm, true);

    std::cout << "create bottom wall: " << bottom_wall << std::endl;

    gran_sys.SetBDFixed(true);

    double cor_p = 0.9;
    double cor_w = 0.5;
    double youngs_modulus = 7e9;  // 70Mpa = 7e7Pa = 7e8 g/(cms^2)
    double mu_s2s = 0.16;
    double mu_s2w = 0.45;
    double mu_roll = 0.2;
    double poisson_ratio = 0.24;

    gran_sys.UseMaterialBasedModel(true);

    gran_sys.SetYoungModulus_SPH(youngs_modulus);
    gran_sys.SetYoungModulus_WALL(youngs_modulus);
    gran_sys.SetYoungModulus_MESH(youngs_modulus);

    gran_sys.SetRestitution_SPH(cor_p);
    gran_sys.SetRestitution_WALL(cor_w);
    gran_sys.SetRestitution_MESH(cor_w);

    gran_sys.SetPoissonRatio_SPH(poisson_ratio);
    gran_sys.SetPoissonRatio_WALL(poisson_ratio);
    gran_sys.SetPoissonRatio_MESH(poisson_ratio);

    // gran_sys.UseMaterialBasedModel(false);

    // double kn = 1e7;
    // double gn = 2e4;
    // double kt = 2e6;
    // double gt = 50;

    // gran_sys.SetKn_SPH2SPH(kn);
    // gran_sys.SetKn_SPH2WALL(kn);
    // gran_sys.SetKn_SPH2MESH(kn);

    // gran_sys.SetGn_SPH2SPH(gn);
    // gran_sys.SetGn_SPH2WALL(gn);
    // gran_sys.SetGn_SPH2MESH(gn);

    // gran_sys.SetKt_SPH2SPH(kt);
    // gran_sys.SetKt_SPH2WALL(kt);
    // gran_sys.SetKt_SPH2MESH(kt);

    // gran_sys.SetGt_SPH2SPH(gt);
    // gran_sys.SetGt_SPH2WALL(gt);
    // gran_sys.SetGt_SPH2MESH(gt);

    // gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    // gran_sys.set_Adhesion_ratio_S2M(params.adhesion_ratio_s2m);
    // gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);

    gran_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    gran_sys.SetFixedStepSize(step_size);
    gran_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
    gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::FORWARD_EULER);
    gran_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s2s);
    gran_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s2w);
    gran_sys.SetStaticFrictionCoeff_SPH2MESH(mu_s2s);

    gran_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gran_sys.SetRollingCoeff_SPH2SPH(mu_roll);
    gran_sys.SetRollingCoeff_SPH2WALL(mu_roll);
    gran_sys.SetRollingCoeff_SPH2MESH(mu_roll);

    std::string mesh_filename(GetChronoDataFile("models/sphere.obj"));

    float ball_radius = 5.0f;
    float initial_ball_pos = initial_surface + initial_height + ball_radius;
    std::vector<float3> mesh_translations(1, make_float3(0.f, 0.f, 0.f));
    std::vector<ChMatrix33<float>> mesh_rotscales(1, ChMatrix33<float>(ball_radius));
    float ball_mass = 1000.0f;

    auto ball_mesh_id = gran_sys.AddMesh(mesh_filename, ChVector<float>(0), ChMatrix33<float>(ball_radius), ball_mass);

    gran_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);
    std::string out_dir = GetChronoOutputPath() + "balldrop_impact_mesh/";
    filesystem::create_directory(filesystem::path(out_dir));

    gran_sys.SetParticleOutputFlags(ABSV);
    gran_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);

    gran_sys.EnableMeshCollision(true);
    gran_sys.Initialize();

    // Create rigid ball_body simulation
    ChSystemSMC sys_ball;
    sys_ball.SetContactForceModel(
        ChSystemSMC::ContactForceModel::Hertz);  // this need to be consistent with the granular bed
    sys_ball.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT);
    sys_ball.Set_G_acc(ChVector<>(grav_X, grav_Y, grav_Z));

    double inertia = 2.0 / 5.0 * ball_mass * ball_radius * ball_radius;
    ChVector<> ball_initial_pos(0, 0, initial_ball_pos);

    std::shared_ptr<ChBody> ball_body(sys_ball.NewBody());
    ball_body->SetMass(ball_mass);
    ball_body->SetInertiaXX(ChVector<>(inertia, inertia, inertia));
    ball_body->SetPos(ball_initial_pos);
    auto sph = chrono_types::make_shared<ChSphereShape>();
    sph->GetSphereGeometry().rad = ball_radius;
    ball_body->AddAsset(sph);
    sys_ball.AddBody(ball_body);

    int currframe = 0;
    unsigned int curr_step = 0;
    int frame_output_freq = 100;  // write position info every 100 time steps
    int cout_freq = 1000;         // cout info every 10 time steps

    clock_t start = std::clock();

    // FILE * pFile;
    // char filename[100];
    // sprintf(filename, "result_meshsphere_%s", argv[1]);
    // pFile = fopen(filename, "w");

    for (double t = 0; t < time_end; t += step_size, curr_step++) {
        gran_sys.ApplyMeshMotion(0, ball_body->GetPos(), ball_body->GetRot(), ball_body->GetPos_dt(),
                                 ball_body->GetWvel_par());

        ChVector<> ball_force;
        ChVector<> ball_torque;
        gran_sys.CollectMeshContactForces(0, ball_force, ball_torque);

        ball_body->Empty_forces_accumulators();
        ball_body->Accumulate_force(ball_force, ball_body->GetPos(), false);
        ball_body->Accumulate_torque(ball_torque, false);

        gran_sys.AdvanceSimulation(step_size);
        sys_ball.DoStepDynamics(step_size);

        // float KE = getSystemKE(sphere_density, sphere_radius, gran_sys, body_points.size());

        // double ball_pos_z = ball_body->GetPos().z();
        // float penetration_d = initial_surface - ball_pos_z + ball_radius;
        // float KE_threshold = 0.1f;
        // if (curr_step % cout_freq == 0){
        //     fprintf(pFile, "%e, %e, %e, %e\n", t, ball_body->GetPos().z(), ball_body->GetPos_dt().z() * 0.01f,
        //     ball_body->GetPos_dtdt().z() * 0.01f);
        // }
    }
    // fclose(pFile);

    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;

    // delete[] meshPosRot;
    // delete[] meshVel;

    return 0;
}
