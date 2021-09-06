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
// Granular material settling in a cylindrical container to generate the bed for ball
// drop test, once settled positions are written
// =============================================================================

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

#include "chrono/core/ChGlobal.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChForce.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/timestepper/ChTimestepper.h"

#include "chrono_thirdparty/filesystem/path.h"

#include "chrono_gpu/ChGpuData.h"
#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_gpu/utils/ChGpuJsonParser.h"

#include "GpuDemoUtils.hpp"

using namespace chrono;
using namespace chrono::gpu;

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file>" << std::endl;
    std::cout << "OR " + name + " <json_file> " + " <input_file> " << std::endl;
}

// calculate kinetic energy of the system
float getSystemKE(double sphere_denisty, double sphere_radius, ChSystemGpu& gpu_sys, int nb) {
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

void findVeloViolation(std::vector<int>& index, ChSystemGpu& gpu_sys, int nb) {
    ChVector<float> velo(0.0f, 0.0f, 0.0f);
    float velo_mag = 0.0f;
    index.clear();
    for (int i = 0; i < nb; i++) {
        velo = gpu_sys.GetParticleVelocity(i);
        velo_mag = velo.Length2();
        if (std::sqrt(velo_mag) > 0.095f) {
            index.push_back(i);
        }
    }
}

// calculate root mean square of the velocity and maximum velocity
void calc_settling_criteria(ChSystemGpu& gpu_sys, int nb, double& rms, double& maxVelo, int& maxVeloID) {
    ChVector<float> velo(0.0f, 0.0f, 0.0f);
    float sum = 0;
    maxVelo = 0;
    float velo_mag = 0.0f;
    maxVeloID = 0;

    for (int i = 0; i < nb; i++) {
        velo = gpu_sys.GetParticleVelocity(i);
        velo_mag = velo.Length2();
        sum = sum + velo_mag;

        if (velo_mag > maxVelo) {
            maxVelo = velo_mag;
            maxVeloID = i;
        }
    }
    rms = std::sqrt(sum / nb);
    maxVelo = std::sqrt(maxVelo);
}

double getBulkDensity(ChSystemGpu& gpu_sys,
                      double sphere_radius,
                      double sphere_density,
                      double bottom,
                      double cyl_radius,
                      int nb) {
    double max_z = gpu_sys.GetMaxParticleZ();

    double height = max_z + sphere_radius - bottom;
    double volume = CH_C_PI * cyl_radius * cyl_radius * height;

    double volume_per_particle = 4.0 / 3.0 * CH_C_PI * std::pow(sphere_radius, 3);
    double mass_per_particle = volume_per_particle * sphere_density;
    double bulk_mass = nb * mass_per_particle;
    double bulk_density = bulk_mass / volume;
    return bulk_density;
}

int main(int argc, char* argv[]) {
    // unit gcm
    double sphere_radius = 0.5f;  // 1cm diameter
    double sphere_density = 2.48;
    double box_X = 31.5;
    double box_Y = 31.5;
    double box_Z = 30.0;

    double grav_X = 0.0f;
    double grav_Y = 0.0f;
    double grav_Z = -980.0f;

    bool useCheckpointInput = false;

    if ((argc != 2 && argc != 1)) {
        ShowUsage(argv[0]);
        return 1;
    }
    if (argc == 2) {
        useCheckpointInput = true;
    }

    // float step_size = 1e-6;
    float step_size = 1e-6;

    float time_end = 5.0f;

    ChSystemGpu gran_sys(sphere_radius, sphere_density, ChVector<float>(box_X, box_Y, box_Z));

    // create cylinder containter
    ChVector<float> cyl_center(0.0f, 0.0f, 0.0f);
    float cyl_rad = std::min(box_X, box_Y) / 2.0f;
    size_t cyl_id = gran_sys.CreateBCCylinderZ(cyl_center, cyl_rad, false, true);

    // declare particle positon vector
    std::vector<chrono::ChVector<float>> body_points;
    // if (useCheckpointInput == true){
    //     body_points = loadPositionCheckpoint<float>(argv[1]);
    // 	std::cout << "reading position input success from " << argv[1] <<std::endl;
    // }
    // else
    // {
    utils::HCPSampler<float> sampler(2.001 * sphere_radius);
    ChVector<float> center(0.0f, 0.0f, 0.0f);
    body_points = sampler.SampleCylinderZ(center, cyl_rad - 5 * sphere_radius, box_Z / 2 - sphere_radius);  // }

    int numSpheres = body_points.size();
    std::cout << "numbers of particles created: " << numSpheres << std::endl;

    // create initial velocity vector
    std::vector<ChVector<float>> initialVelo;
    // randomize initial velocity to e
    double velo_mag = 100;
    for (int i = 0; i < numSpheres; i++) {
        ChVector<float> rand_velo(ChRandom() - 0.5, ChRandom() - 0.5, ChRandom() - 0.5);
        rand_velo.Normalize();
        initialVelo.push_back(rand_velo * velo_mag);
    }

    // assign initial position and velocity to the granular system
    gran_sys.SetParticles(body_points, initialVelo);

    gran_sys.SetBDFixed(true);

    double cor_p = 0.9;
    double cor_w = 0.5;
    double youngs_modulus = 7e9;  // 70Mpa = 7e7Pa = 7e8 g/(cms^2)
    double mu_s2s = 0.16;
    double mu_s2w = 0.45;
    double mu_roll = 0.09;
    double poisson_ratio = 0.24;

    gran_sys.UseMaterialBasedModel(true);

    gran_sys.SetYoungModulus_SPH(youngs_modulus);
    gran_sys.SetYoungModulus_WALL(youngs_modulus);

    gran_sys.SetRestitution_SPH(cor_p);
    gran_sys.SetRestitution_WALL(cor_w);

    gran_sys.SetPoissonRatio_SPH(poisson_ratio);
    gran_sys.SetPoissonRatio_WALL(poisson_ratio);

    gran_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

    gran_sys.SetPsiFactors(32.0f, 16.0f);

    gran_sys.SetRollingMode(CHGPU_ROLLING_MODE::SCHWARTZ);
    gran_sys.SetRollingCoeff_SPH2SPH(mu_roll);
    gran_sys.SetRollingCoeff_SPH2WALL(mu_roll);

    gran_sys.SetFixedStepSize(step_size);
    gran_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
    gran_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::FORWARD_EULER);
    gran_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s2s);
    gran_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s2w);

    gran_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);
    std::string out_dir = GetChronoOutputPath() + "balldrop_settling_smallerSpacing/";
    filesystem::create_directory(filesystem::path(out_dir));
    gran_sys.SetParticleOutputFlags(ABSV);
    gran_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);
    gran_sys.SetRecordingContactInfo(true);

    gran_sys.Initialize();

    unsigned int out_fps = 10000;
    std::cout << "Rendering at " << out_fps << "FPS" << std::endl;

    unsigned int out_steps = (unsigned int)(1.0 / (out_fps * step_size));

    clock_t start = std::clock();
    double t = 0.0f;
    int curr_step = 0;
    double frame_step = 0.01;
    double root_mean_squ = 0.0f;
    ;
    double maxVelo = 0.0f;
    float bulk_density;
    int maxVeloID = 0;

    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> angVelo;
    ChVector<float> acc;
    ChVector<float> normalF;
    ChVector<float> slidingFr;
    // ChVector<float> rollingTr;

    double t_1 = 4.0f;
    while (t < t_1) {
        gran_sys.AdvanceSimulation(step_size);
        if (curr_step % 1000 == 0) {
            std::cout << t << std::endl;
        }
        curr_step = curr_step + 1;
        t = t + step_size;
    }
    std::vector<int> index;
    findVeloViolation(index, gran_sys, numSpheres);

    double t_end = t_1 + 1.0f;
    std::vector<FILE*> filePtrArray;
    for (int ii = 0; ii < index.size(); ii++) {
        int pID = index[ii];
        char fileName[100];
        sprintf(fileName, "%s/particle_%05d_info.csv", out_dir.c_str(), pID);
        FILE* filePtr = fopen(fileName, "w");

        filePtrArray.push_back(filePtr);
        std::cout << "create file for particle number " << pID << std::endl;
    }

    while (t < t_end) {
        gran_sys.AdvanceSimulation(step_size);

        for (int ii = 0; ii < index.size(); ii++) {
            int myID = index[ii];
            pos = gran_sys.GetParticlePosition(myID);
            velo = gran_sys.GetParticleVelocity(myID);
            angVelo = gran_sys.GetParticleAngVelocity(myID);
            acc = gran_sys.GetParticleLinAcc(myID);
            fprintf(filePtrArray.at(ii), "%e, %e, %e, %e, %e, %e, %e, %e, %e, %e, ", t, pos.x(), pos.y(), pos.z(),
                    velo.x(), velo.y(), velo.z(), acc.x(), acc.y(), acc.z());

            std::vector<int> neighborList;
            gran_sys.getNeighbors(myID, neighborList);

            for (int jj = 0; jj < neighborList.size(); jj++) {
                int theirID = neighborList.at(jj);
                normalF = gran_sys.getNormalForce(myID, theirID);
                slidingFr = gran_sys.getSlidingFrictionForce(myID, theirID);
                fprintf(filePtrArray.at(ii), "%d, %e, %e, %e, %e, %e, %e, ", theirID, normalF.x(), normalF.y(),
                        normalF.z(), slidingFr.x(), slidingFr.y(), slidingFr.z());
            }
            fprintf(filePtrArray.at(ii), "\n");
        }
        curr_step = curr_step + 1;
        t = t + step_size;

        if (curr_step % 10000 == 0) {
            calc_settling_criteria(gran_sys, numSpheres, root_mean_squ, maxVelo, maxVeloID);
            bulk_density =
                getBulkDensity(gran_sys, sphere_radius, sphere_density, -box_Z / 2.0f, box_X / 2.0f, numSpheres);

            printf("%e, %e, %e, %d, %e\n", t, root_mean_squ, maxVelo, maxVeloID, bulk_density);

            char filename[100];
            sprintf(filename, "%s/step%06d.csv", out_dir.c_str(), (int)(curr_step / 10000));
            gran_sys.WriteParticleFile(std::string(filename));
        }
    }

    //        if (curr_step%100000 == 0){

    // calc_settling_criteria(gran_sys, numSpheres, root_mean_squ, maxVelo, maxVeloID);
    // bulk_density = getBulkDensity(gran_sys, sphere_radius, sphere_density, -box_Z/2.0f, box_X/2.0f, numSpheres);

    // printf("%e, %e, %e, %d, %e\n", t, root_mean_squ, maxVelo, maxVeloID, bulk_density);
    // if (curr_step%1000000 == 0){
    //     char filename[100];
    //     sprintf(filename, "%s/step%06d.csv", out_dir.c_str(), (int)(curr_step/1000000));
    //     gran_sys.WriteParticleFile(std::string(filename));
    // }

    //        }

    // clock_t end = std::clock();
    // double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    // fclose(file_0);
    // fclose(file_1);
    // fclose(file_2);

    // std::cout << "Time: " << total_time << " seconds" << std::endl;

    // delete[] meshPosRot;
    // delete[] meshVel;

    return 0;
}
