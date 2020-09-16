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
// Chrono::Granular simulation of granular material settled in cylinder first 
// then pushed from the top, input can be user defined json file or material based
// property from Arman et al 2017
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
float L_CGS_TO_SI = 1e-2;

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
void printReactionForces(ChSystemGranularSMC &sys, size_t bc_id){
    float reaction_force[3];
    bool success = sys.getBCReactionForces(bc_id, reaction_force);
    if (!success) {
        std::cout << "ERROR! can not get reaction force on plate " << bc_id;
    }
    else{
        std::cout << "top plate " << reaction_force[0] * F_CGS_TO_SI << ", " << reaction_force[1] * F_CGS_TO_SI << ", " << reaction_force[2] * F_CGS_TO_SI << std::endl;
    }
}

// print out reaction froce into a file
void printReactionForces(ChSystemGranularSMC &sys, size_t bc_id, ChStreamOutAsciiFile &stream ){
    float reaction_force[3];
    bool success = sys.getBCReactionForces(bc_id, reaction_force);
    if (!success) {
        std::cout << "ERROR! can not get reaction force on plate " << bc_id;
    }
    else{
        double F_mag = std::sqrt(std::pow(reaction_force[0],2) + std::pow(reaction_force[1],2) + std::pow(reaction_force[2],2));
        stream << ", "  << F_mag * F_CGS_TO_SI; 
    }
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

// evaluate void ratio of the material
// TODO here volume is the cylinder
// float calculateVoidRatio(ChSystemGranularSMC &sys, std::vector<size_t> wall_list){
//     int numSpheres = sys.getNumSpheres();
//     float radius = sys.getRadius();
//     float sphereVol = 4.0/3.0 * CH_C_PI * pow(radius,3) * numSpheres;

//     float boxVol = 1.0f;
//     float dim;
//     float3 platePos_1, platePos_2;
//     for (unsigned int i = 0; i < 3; i++){
//         platePos_1 = sys.Get_BC_Plane_Position(wall_list[2*i]);
//         platePos_2 = sys.Get_BC_Plane_Position(wall_list[2*i+1]);
//         dim = Length(platePos_1 - platePos_2);
//         boxVol = boxVol * dim;
//     }
//     return (1.0f - sphereVol/boxVol);
// }

// evaluate stress on the cylinder wall
float calculateCylWallStress(size_t cyl_id, float cyl_rad, float cyl_height, ChSystemGranularSMC &gran_sys){
    float reaction_force[3];
    gran_sys.getBCReactionForces(cyl_id, reaction_force);
    float force_mag = std::sqrt(reaction_force[0] * reaction_force[0] + reaction_force[1] * reaction_force[1]) * F_CGS_TO_SI;
    float cyl_surface = 2 * CH_C_PI * cyl_rad * cyl_height * L_CGS_TO_SI * L_CGS_TO_SI;
    float stress = force_mag/cyl_surface;
    return stress;
}

enum RUN_MODE {MATERIAL_BASED_MU_R = 0, MATERIAL_BASED_NO_ROLL = 1, USER_DEFINED_MU_R = 2, USER_DEFINED_NO_ROLL = 3};

int main(int argc, char* argv[]) {

    sim_param_holder params;
    if (argc != 3 || ParseJSON(argv[1], params) == false) {
        ShowUsage(argv[0]);
        return 1;
    }

    RUN_MODE run_mode = (RUN_MODE)std::atoi(argv[2]);

    // material based parameter
    float rho = 7.8;
    float YoungsModulus = 2e9;
    float nu = 0.28;
    float COR = 0.6;


    // Setup simulation
    ChSystemGranularSMC gran_sys(params.sphere_radius, rho,
                                 make_float3(params.box_X, params.box_Y, params.box_Z));

    ChGranularSMC_API apiSMC;

    checkStepSize(&params);
    apiSMC.setGranSystem(&gran_sys);

    // cylinder boundary
    float cyl_center[3] = {0.0f, 0.0f, 0.0f};
    float cyl_height = params.box_Z;
    float cyl_rad = std::min(params.box_X, params.box_Y)/2.0f;
    size_t cyl_id = gran_sys.Create_BC_Cyl_Z(cyl_center, cyl_rad, false, true);

    // sampler
    utils::HCPSampler<float> sampler(2.1 * params.sphere_radius); //wtf is 2.1 for diameter??
    std::vector<ChVector<float>> initialPos;

    // randomize by layer
    ChVector<float> center(0.0f, 0.0f, -params.box_Z/2 + params.sphere_radius);
    // fill up each layer
    while (center.z() + params.sphere_radius < params.box_Z/2 * 0.5){
        auto points = sampler.SampleCylinderZ(center, cyl_rad, 0);
        initialPos.insert(initialPos.end(), points.begin(), points.end());
        center.z() += 2.1 * params.sphere_radius;
    }

    int numSpheres = initialPos.size();
    std::cout << "number of spheres: " << numSpheres;
    
    apiSMC.setElemsPositions(initialPos);

    gran_sys.setPsiFactors(params.psi_T, params.psi_L);

    float mu_s_s2s = 0.096;
    float mu_s_s2w = 0.28;

    float sigma = (1 - std::pow(nu,2))/YoungsModulus;
    float Kn_s2s = 4.0f/(6.0f * sigma) * std::sqrt(params.sphere_radius/2.0f) * std::sqrt(params.sphere_radius);
    float Kn_s2w = 4.0f/(6.0f * sigma) * params.sphere_radius;

    printf("material based property: Kn_s2s = %f, Kn_s2w = %f\n", Kn_s2s, Kn_s2w);

    // normal force model
    gran_sys.set_K_n_SPH2SPH(Kn_s2s);
    gran_sys.set_K_n_SPH2WALL(Kn_s2w);
    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);
    //tangential force model
    gran_sys.set_K_t_SPH2SPH(2.0f/7.0f * Kn_s2s);
    gran_sys.set_K_t_SPH2WALL(2.0f/7.0f * Kn_s2w);
    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
    gran_sys.set_static_friction_coeff_SPH2SPH(mu_s_s2s);
    gran_sys.set_static_friction_coeff_SPH2WALL(mu_s_s2w);

    // set rolling friction model
    // apply rolling friction and see what happens
    float mu_r = 0.01;
    gran_sys.set_rolling_mode(GRAN_ROLLING_MODE::SCHWARTZ);
    gran_sys.set_rolling_coeff_SPH2SPH(mu_r);
    gran_sys.set_rolling_coeff_SPH2WALL(0.0f);
    gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);
    gran_sys.setOutputMode(params.write_mode);
    filesystem::create_directory(filesystem::path(params.output_dir));

    // Set the position of the BD
    gran_sys.set_BD_Fixed(true);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::FORWARD_EULER);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_fixed_stepSize(params.step_size);

    gran_sys.setVerbose(params.verbose);
//    gran_sys.setRecordingContactInfo(true);

    float topWallPos[3]    = {0, 0,  (float)(params.box_Z/2.0)};
    float topWallN[3]    = {0,0,-1.0};
    size_t topWall    = gran_sys.Create_BC_Plane(topWallPos,    topWallN,    true);

    float topWall_vel;
    // i would like it to start from an offset 
    float topWall_offset;
    float topWall_moveTime;
    std::function<double3(float)> topWall_posFunc = [&topWall_offset, &topWall_vel, &topWall_moveTime](float t) {
        double3 pos = {0, 0, 0};
        pos.z =  topWall_offset + topWall_vel * (t - topWall_moveTime);
        return pos;
    };

    gran_sys.setOutputFlags(GRAN_OUTPUT_FLAGS::ABSV | GRAN_OUTPUT_FLAGS::ANG_VEL_COMPONENTS);

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

    float cyl_reaction_force[3];
    float plane_reaction_force[3];
     
    while (sysKE > KE_settle_threshold) {

        printf("rendering frame %u, curr_time = %.4f, ", currframe, curr_time);
        char filename[100];
        sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe);
        gran_sys.writeFile(std::string(filename));
        gran_sys.advance_simulation(frame_step);
        curr_time += frame_step;

        platePos = gran_sys.Get_BC_Plane_Position(topWall);
        std::cout << "plate position z : " << platePos.z;

        sysKE = getSystemKE(params, apiSMC, numSpheres) * KE_CGS_TO_SI;
        std::cout << ", system KE: " << sysKE;

        
        nc = gran_sys.getNumContacts();
        std::cout << ", nc: " << nc;

        gran_sys.getBCReactionForces(cyl_id, cyl_reaction_force);
        std::cout << ", cyl wall " << cyl_reaction_force[0] * F_CGS_TO_SI << ", " << cyl_reaction_force[1] * F_CGS_TO_SI << ", " << cyl_reaction_force[2] * F_CGS_TO_SI;

        std::cout << "\n";
        currframe++;
    }

    topWall_vel = -1;
    // i would like it to start from the top most sphere
    topWall_offset = gran_sys.get_max_z() + 2 * params.sphere_radius - topWallPos[2];
    topWall_moveTime = curr_time;

    // sphere settled now push the plate downward
    gran_sys.set_BC_offset_function(topWall, topWall_posFunc);

    // continue simulation until the end
    while (curr_time < params.time_end){
        printf("rendering frame %u, curr_time = %.4f, ", currframe, curr_time);
        char filename[100];
        sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe);
        gran_sys.writeFile(std::string(filename));
        gran_sys.advance_simulation(frame_step);
        stream << curr_time;
        curr_time += frame_step;

        platePos = gran_sys.Get_BC_Plane_Position(topWall);
        std::cout << "plate position z : " << platePos.z;
        stream << ", " << platePos.z;

        sysKE = getSystemKE(params, apiSMC, numSpheres) * KE_CGS_TO_SI;
        std::cout << ", system KE: " << sysKE;
        stream << ", " << sysKE;
        
        nc = gran_sys.getNumContacts();
        std::cout << ", nc: " << nc;
        stream << ", " << nc;

        float cyl_height = platePos.z + params.box_Z/2.0f;
        float cyl_stress = calculateCylWallStress(cyl_id, cyl_rad, cyl_height, gran_sys);

        gran_sys.getBCReactionForces(cyl_id, cyl_reaction_force);
        std::cout << ", cyl wall " << cyl_reaction_force[0] * F_CGS_TO_SI << ", " << cyl_reaction_force[1] * F_CGS_TO_SI << ", " << cyl_reaction_force[2] * F_CGS_TO_SI;

        gran_sys.getBCReactionForces(topWall, plane_reaction_force);
        float cyl_cross_section = CH_C_PI * std::pow(cyl_rad * L_CGS_TO_SI, 2);
        float topWall_stress = plane_reaction_force[2] * F_CGS_TO_SI/cyl_cross_section;
        std::cout << ", top wall stress " << topWall_stress;

        std::cout << "\n";
        stream << "\n";

        currframe++;

    }


    // sprintf(contactFilename, "%s/contact%06d", params.output_dir.c_str(), currframe-1);
    // gran_sys.writeContactInfoFile(std::string(contactFilename));
    // printf("successfully writing contact info file!\n");
    // stream.GetFstream().flush();
    return 0;
}