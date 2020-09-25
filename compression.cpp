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
// Chrono::Granular simulation of a box of granular material which
// is first let to settle and then compressed by advancing the top wall
// into the material.
// =============================================================================

#include <iostream>
#include <string>
#include <cmath>
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono_granular/utils/ChCudaMathUtils.cuh"
#include "demos/granular/ChGranularDemoUtils.hpp"
#include "chrono/core/ChStream.h"
#include "chrono/core/ChVector.h"

using namespace chrono;
using namespace chrono::granular;

// unit conversion from cgs to si
float F_CGS_TO_SI = 1e-5;
float KE_CGS_TO_SI = 1e-7;

// Show command line usage
void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file>" + " <input position file> " << std::endl;
    std::cout << "OR " + name + " <json_file>" << std::endl;
}

// check step size, delta t ~ sqrt(m/k)
void checkStepSize(sim_param_holder* params){
    float volume = 4.0/3.0 * CH_C_PI * std::pow(params->sphere_radius, 3);
    float mass = volume * params->sphere_density;
    float freq = std::sqrt(params->normalStiffS2S/mass);
    float crit_dt = 1.0/freq;
    if ( params->step_size < crit_dt){
        std::cout << "critical step size is " << crit_dt << ", step size is safe\n";
    }
    else{
        std::cout << "critical step size is  " << crit_dt << ", step size is not safe\n";
    }

}

// print out reaction force applied on the top plate onto std
void printReactionForces(ChSystemGranularSMC &sys, std::vector<size_t> bc_id_list){
    float reaction_force[3] = {0.0, 0.0, 0.0};
    for (unsigned int i = 0; i < bc_id_list.size(); i++){
        bool success = sys.getBCReactionForces(bc_id_list[i], reaction_force);
        if (!success) {
            std::cout << "ERROR! can not get reaction force on plate " << bc_id_list[i];
        }
        else{
            std::cout << "plate " << i << ", "  << reaction_force[0] * F_CGS_TO_SI << ", " << reaction_force[1] * F_CGS_TO_SI << ", " << reaction_force[2] * F_CGS_TO_SI << std::endl;
        }
    }
}

// print out reaction froce into a file
void printReactionForces(ChSystemGranularSMC &sys, std::vector<size_t> bc_id_list, ChStreamOutAsciiFile &stream ){
    float reaction_force[3] = {0.0, 0.0, 0.0};
    float F_mag;
    for (unsigned int i = 0; i < bc_id_list.size(); i++){
        bool success = sys.getBCReactionForces(bc_id_list[i], reaction_force);
        if (!success) {
            std::cout << "ERROR! can not get reaction force on plate " << bc_id_list[i];
        }
        else{
            F_mag = std::sqrt(std::pow(reaction_force[0],2) + std::pow(reaction_force[1],2) + std::pow(reaction_force[2],2));
            stream << ", " << i << ", "  << F_mag * F_CGS_TO_SI; 
        }
    }
}

// evaluate void ratio of the material
float calculateVoidRatio(ChSystemGranularSMC &sys, std::vector<size_t> wall_list){
    int numSpheres = sys.getNumSpheres();
    float radius = sys.getRadius();
    float sphereVol = 4.0/3.0 * CH_C_PI * pow(radius,3) * numSpheres;

    float boxVol = 1.0f;
    float dim;
    float3 platePos_1, platePos_2;
    for (unsigned int i = 0; i < 3; i++){
        platePos_1 = sys.Get_BC_Plane_Position(wall_list[2*i]);
        platePos_2 = sys.Get_BC_Plane_Position(wall_list[2*i+1]);
        dim = Length(platePos_1 - platePos_2);
        boxVol = boxVol * dim;
    }
    return (1.0f - sphereVol/boxVol);
}

// initialize sphere position by PD sampler, return total number of spheres
int randomizeSpherePosition(sim_param_holder &params, ChGranularSMC_API &api_sys){
    float eps = 0.97;   
    // half box dimension
    ChVector<float> hdims((float)(params.box_X / 2.0 * eps), 
                          (float)(params.box_Y / 2.0 * eps),
                          (float)(params.box_Z / 2.0 * eps));
    ChVector<float> center(0.f, 0.f, 0.0f);

    // Fill box layer by layer 
    std::vector<ChVector<float>> body_points =
        utils::PDLayerSampler_BOX<float>(center, hdims, 2.f * params.sphere_radius, 1.05f);

    // fill big domain with bodies by samplers
    api_sys.setElemsPositions(body_points);
    
    // total number of spheres
    int numSpheres = body_points.size();
    return numSpheres;
}


int main(int argc, char* argv[]) {

    sim_param_holder params;
    if ((argc != 2 && argc != 3) || ParseJSON(argv[1], params) == false) {
        ShowUsage(argv[0]);
        return 1;
    }

    // declare input position variable
    std::vector<ChVector<float>> initialPos;

    // user did not specify position input, randomize position input with PD Sampler
    if (argc == 2){
        ChVector<float> hdims((float)(params.box_X / 2.0), 
                              (float)(params.box_Y / 2.0),
                              (float)(params.box_Z / 2.0 - params.sphere_radius));
        ChVector<float> center(0.f, 0.f, 0.0f);

        // Fill box layer by layer 
        initialPos = utils::PDLayerSampler_BOX<float>(center, hdims, 2.f * params.sphere_radius, 1.05f);
    }
    else {
        initialPos = loadPositionCheckpoint<float>(argv[2]);
    }

    // Setup simulation
    ChSystemGranularSMC gran_sys(params.sphere_radius, params.sphere_density,
                                 make_float3(params.box_X, params.box_Y, params.box_Z));

    ChGranularSMC_API apiSMC;

    checkStepSize(&params);
    apiSMC.setGranSystem(&gran_sys);
    
    std::cout << "number of spheres: " << initialPos.size();

    gran_sys.setPsiFactors(params.psi_T, params.psi_L);

    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);

    gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
    gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
    gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
    gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);

    // set rolling friction model
    // apply rolling friction and see what happens
    float mu_r = 0.1;
    gran_sys.set_rolling_mode(GRAN_ROLLING_MODE::SCHWARTZ);
    gran_sys.set_rolling_coeff_SPH2SPH(mu_r);
    gran_sys.set_rolling_coeff_SPH2WALL(0.0f);
    gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);
    gran_sys.setOutputMode(params.write_mode);
    filesystem::create_directory(filesystem::path(params.output_dir));


    // fill box with spheres
    apiSMC.setElemsPositions(initialPos);

    // Set the position of the BD
    gran_sys.set_BD_Fixed(true);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::FORWARD_EULER);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_fixed_stepSize(params.step_size);

    gran_sys.setVerbose(params.verbose);
    gran_sys.setRecordingContactInfo(true);

    // create box wall pos z axis point up, x axis point out of the screen
    float frontWallPos[3]  = { (float)(params.box_X/2.0), 0, 0};
    float backWallPos[3]   = {-(float)(params.box_X/2.0), 0, 0};
    float rightWallPos[3]  = {0,  (float)(params.box_Y/2.0), 0};
    float leftWallPos[3]   = {0, -(float)(params.box_Y/2.0), 0};
    float topWallPos[3]    = {0, 0,  (float)(params.box_Z/2.0)};
    float bottomWallPos[3] = {0, 0, -(float)(params.box_Z/2.0)};

    // plate normal 
    float frontWallN[3]  = {-1.0,0,0};
    float backWallN[3]   = { 1.0,0,0};
    float rightWallN[3]  = {0,-1.0,0};
    float leftWallN[3]   = {0, 1.0,0};
    float topWallN[3]    = {0,0,-1.0};
    float bottomWallN[3] = {0,0, 1.0};

    size_t frontWall  = gran_sys.Create_BC_Plane(frontWallPos,  frontWallN,  true);
    size_t backWall   = gran_sys.Create_BC_Plane(backWallPos,   backWallN,   true);
    size_t rightWall  = gran_sys.Create_BC_Plane(rightWallPos,  rightWallN,  true);
    size_t leftWall   = gran_sys.Create_BC_Plane(leftWallPos,   leftWallN,   true);
    size_t topWall    = gran_sys.Create_BC_Plane(topWallPos,    topWallN,    true);
    size_t bottomWall = gran_sys.Create_BC_Plane(bottomWallPos, bottomWallN, true);

    std::vector<size_t> wall_list;
    wall_list.push_back(frontWall);
    wall_list.push_back(backWall);
    wall_list.push_back(leftWall);
    wall_list.push_back(rightWall);
    wall_list.push_back(bottomWall);
    wall_list.push_back(topWall);

    float topWall_vel;
    // i would like it to start from 10cm 
    float topWall_offset;
    float topWall_moveTime;
    std::function<double3(float)> topWall_posFunc = [&topWall_offset, &topWall_vel, &topWall_moveTime](float t) {
        double3 pos = {0, 0, 0};
        pos.z =  topWall_offset + topWall_vel * (t - topWall_moveTime);
        return pos;
    };

    std::function<double3(float)> topWall_posFunc_stop = [&params](float t) {
        double3 pos = {0,0,0};

        // move at -0.2 m/s
        pos.z = -params.box_Z * 0.5f + (-0.2) * 2;

        return pos;

    };

    gran_sys.setOutputFlags(GRAN_OUTPUT_FLAGS::ABSV);



    gran_sys.initialize();
    
    // output frames per second
    int fps = 100;
    // assume we run for at least one frame
    float frame_step = 1.0f / fps;
    float curr_time = 0;
    int currframe = 0;

    char filename[100];
    sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe);
    gran_sys.writeFile(std::string(filename));

    std::cout << "frame step is " << frame_step << std::endl;

    char contactFilename[100];
    // Run settling experiments

    std::string logName = "triaxial_logging_smallerRadius";
    ChStreamOutAsciiFile stream(logName.c_str());
    stream.SetNumFormat("%.12g");
    stream << "time, KE " << "\n";

    // initialize values that I want to keep track of
    float KE_settle_threshold = 1E-3;
    float sysKE = 1E8;
    int nc; // total number of contacts
    float3 platePos;  // top plate position
    float voidRatio; 
    float voidRatio_threshold = 0.2;


//    float sysKE = 
     
    while (sysKE > KE_settle_threshold) {

        printf("rendering frame %u, curr_time = %.4f, ", currframe, curr_time);
        char filename[100];
        sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe);
        gran_sys.writeFile(std::string(filename));
        gran_sys.advance_simulation(frame_step);
        curr_time += frame_step;

        platePos = gran_sys.Get_BC_Plane_Position(topWall);
        std::cout << "plate position z : " << platePos.z;

        sysKE = gran_sys.getSystemKineticEnergy() * KE_CGS_TO_SI;
        std::cout << ", system KE: " << sysKE;

        
        nc = gran_sys.getNumContacts();
        std::cout << ", nc: " << nc;

        voidRatio = calculateVoidRatio(gran_sys, wall_list);
        std::cout << ", void ratio: " << voidRatio;
        std::cout << "\n";
        currframe++;
    }

    topWall_vel = -1;
    // i would like it to start from the top most sphere
    topWall_offset = gran_sys.get_max_z() + 2 * params.sphere_radius - topWallPos[2];
    topWall_moveTime = curr_time;

    // sphere settled now push the plate downward
    gran_sys.set_BC_offset_function(wall_list[wall_list.size()-1], topWall_posFunc);

    // continue simulation until the end
    while (curr_time < params.time_end && voidRatio > voidRatio_threshold){
        printf("rendering frame %u, curr_time = %.4f, ", currframe, curr_time);
        char filename[100];
        sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe);
        gran_sys.writeFile(std::string(filename));
        gran_sys.advance_simulation(frame_step);
        stream << curr_time;
        curr_time += frame_step;

        platePos = gran_sys.Get_BC_Plane_Position(wall_list[wall_list.size()-1]);
        std::cout << "plate position z : " << platePos.z;
        stream << ", " << platePos.z;

        sysKE = gran_sys.getSystemKineticEnergy() * KE_CGS_TO_SI;
        std::cout << ", system KE: " << sysKE;
        stream << ", " << sysKE;
        
        nc = gran_sys.getNumContacts();
        std::cout << ", nc: " << nc;
        stream << ", " << nc;

        printReactionForces(gran_sys, wall_list, stream);
        voidRatio = calculateVoidRatio(gran_sys, wall_list);
        std::cout << ", void ratio: " << voidRatio;
        stream << ", " << voidRatio;

        std::cout << "\n";
        stream << "\n";

        currframe++;

    }


    sprintf(contactFilename, "%s/contact%06d", params.output_dir.c_str(), currframe-1);
    gran_sys.writeContactInfoFile(std::string(contactFilename));
    printf("successfully writing contact info file!\n");
    stream.GetFstream().flush();
    return 0;
}