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
// Low-velocity impact test of a sphere (modelled as analytical boundary condition) dropped on the a bed of settle
// granular material unit: cgs
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
    std::cout << "usage: " + name + " <input_file> " + " <impact velocity (m/s)> " + " <rolling friction mu_r>"
              << std::endl;
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
    double box_Z = 50.0;  // increase box_Z dimension to include sphere mesh

    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -980.0f;

    // TODO: modifiy this to allow checkpointing
    float time_settle = 1.5;  // time to allow the granular material to time_settle
    float time_impact = 0.15;
    float time_end = time_settle + time_impact;
    if (argc != 4) {
        ShowUsage(argv[0]);
        return 1;
    }

    float projectile_velo_cm = atof(argv[2]) * 100;  // convert to cm/s
    float step_size = 1e-6;
    ChSystemGpu gran_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    // declare particle positon vector
    std::vector<chrono::ChVector<float>> body_points;
    char input_pos_filename[100];
    sprintf(input_pos_filename, "input_data/balldrop/%s", argv[1]);

    body_points = loadPositionCheckpoint<float>(input_pos_filename);
    gran_sys.SetParticles(body_points);
    std::cout << "sucessfully load file from: " << input_pos_filename << std::endl;
    std::cout << "number of particles: " << body_points.size() << std::endl;

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

    std::cout << "create bottom plate: " << bottom_wall << std::endl;

    gran_sys.SetBDFixed(true);

    double cor_p = 0.9;
    // double cor_p = 0.5;
    double cor_w = 0.5;
    double youngs_modulus = 7e9;  // 700Mpa = 7e8Pa = 7e9 g/(cms^2)
    double mu_s2s = 0.16;
    double mu_s2w = 0.45;
    double mu_roll = atof(argv[3]);
    double poisson_ratio = 0.24;

    gran_sys.UseMaterialBasedModel(true);

    gran_sys.SetYoungModulus_SPH(youngs_modulus);
    gran_sys.SetYoungModulus_WALL(youngs_modulus);
    gran_sys.SetRestitution_SPH(cor_p);
    gran_sys.SetRestitution_WALL(cor_w);
    gran_sys.SetPoissonRatio_SPH(poisson_ratio);
    gran_sys.SetPoissonRatio_WALL(poisson_ratio);

    gran_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    gran_sys.SetFixedStepSize(step_size);
    gran_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
    gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::FORWARD_EULER);
    gran_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s2s);
    gran_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s2w);

    gran_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gran_sys.SetRollingCoeff_SPH2SPH(mu_roll);
    gran_sys.SetRollingCoeff_SPH2WALL(mu_roll);

    float ball_radius = 5.0f;
    float initial_H = 0.1;

    float initial_ball_pos = box_Z / 2 - ball_radius;
    float ball_mass = 1000;
    ChVector<> ball_initial_pos(0, 0, initial_ball_pos);
    size_t ball_id = gran_sys.CreateBCSphere(ball_initial_pos, ball_radius, true, true, ball_mass);

    gran_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);
    std::string out_dir = GetChronoOutputPath() + "balldrop_impact_bc/";
    filesystem::create_directory(filesystem::path(out_dir));

    gran_sys.SetParticleOutputFlags(ABSV);
    gran_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);
    gran_sys.SetVerbosity(CHGPU_VERBOSITY::QUIET);

    gran_sys.Initialize();
    gran_sys.DisableBCbyID(ball_id);

    int currframe = 0;
    unsigned int curr_step = 0;
    int frame_output_freq = 100;  // write position info every 100 time steps
    int cout_freq = 10;           // cout info every 10 time steps

    ChVector<> bc_pos;
    ChVector<> bc_velo;
    ChVector<float> bc_force(0.0f, 0.0f, 0.0f);

    double ball_pos_contact = 0;  // position of projectile at contact

    float frame_step = 0.0001;
    //    float frame_step = step_size;

    clock_t start = std::clock();

    float frame_step_settle = 0.01;
    float sys_KE;
    for (double t = 0; t < time_settle; t += frame_step_settle) {
        gran_sys.AdvanceSimulation(frame_step);
        float sys_KE = gran_sys.GetParticlesKineticEnergy();
        std::cout << t << ", " << sys_KE << std::endl;
        // char filename[100];
        // sprintf(filename, "%s/settling_step%06d.csv", out_dir.c_str(), currframe++);
        // gran_sys.WriteParticleFile(std::string(filename));
    }

    // find highest point of the top layer particles that are within a certain radius, this is to minimize boundary
    // effect
    double max_z = -1000;
    for (int i = 0; i < body_points.size(); i++) {
        double dist =
            std::sqrt(body_points.at(i).x() * body_points.at(i).x() + body_points.at(i).y() * body_points.at(i).y());

        if (dist < 6 * sphere_radius && body_points.at(i).z() > max_z) {
            max_z = body_points.at(i).z();
        }
    }

    float initial_surface = max_z + sphere_radius;
    std::cout << "highest point at " << initial_surface << "cm" << std::endl;
    float initial_volume = box_X * box_Y * (initial_surface - bottom_loc);
    int numSpheres = body_points.size();
    std::cout << "initialize " << numSpheres << " particles" << std::endl;
    float volume_per_particle = 4.0 / 3.0 * CH_C_PI * std::pow(sphere_radius, 3);
    float mass_per_particle = volume_per_particle * sphere_density;

    float bulk_mass = numSpheres * mass_per_particle;
    // TODO: cout bulk density and packing frac after settling stage
    float bulk_density = bulk_mass / initial_volume;
    float packing_frac = numSpheres * volume_per_particle / initial_volume;
    std::cout << "bulk density is " << bulk_density << "g/cm3" << std::endl;
    std::cout << "packing frac is " << packing_frac << std::endl;

    // set bc sphere position and Velocity
    ChVector<float> projectile_initial_pos(0.0, 0.0, initial_surface + ball_radius);
    ChVector<float> projectile_initial_velocity(0.0, 0.0, -projectile_velo_cm);

    gran_sys.EnableBCbyID(ball_id);
    gran_sys.SetBCSpherePosition(ball_id, projectile_initial_pos);
    gran_sys.SetBCSphereVelocity(ball_id, projectile_initial_velocity);

    std::cout << "projectile position " << projectile_initial_pos.z() << " cm"
              << ", velocity " << projectile_initial_velocity.z() << " cm/s" << std::endl;

    FILE* pFile;
    char filename[100];
    sprintf(filename, "bc_sphere_result_mur_%.2f_velo_%.2fms-1.csv", mu_roll, projectile_velo_cm * 0.01);
    pFile = fopen(filename, "w");

    ////////////////////////////////////contst/////
    ////////// impact stage /////////////////
    /////////////////////////////////////////
    for (double t = time_settle; t < time_end; t += frame_step, curr_step++) {
        gran_sys.AdvanceSimulation(frame_step);
        bc_pos = gran_sys.GetBCSpherePosition(ball_id);
        bc_velo = gran_sys.GetBCSphereVelocity(ball_id);
        gran_sys.GetBCReactionForces(ball_id, bc_force);

        fprintf(pFile, "%e, %e, %e, %e\n", t, bc_pos.z(), bc_velo.z() * 0.01f,
                (bc_force.z() / ball_mass + grav_Z) * 0.01f);
    }

    fclose(pFile);

    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;

    // delete[] meshPosRot;
    // delete[] meshVel;

    return 0;
}
