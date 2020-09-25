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
// Chrono::Granular test of spheres settling in a frictional bucket
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
void checkStepSize(float sphere_radius, float sphere_density, float stiffness, float step_size){
    float volume = 4.0/3.0 * CH_C_PI * std::pow(sphere_radius, 3);
    float mass = volume * sphere_density;
    float freq = std::sqrt(stiffness/mass);
    float crit_dt = 1.0/freq;
    if ( step_size < crit_dt){
        std::cout << "critical step size is " << crit_dt << ", step size is safe\n";
    }
    else{
        std::cout << "critical step size is  " << crit_dt << ", step size is not safe\n";
    }

}


float getMass(float rad, float density){
    float volume = 4.0f/3.0f * CH_C_PI * std::pow(rad, 3);
    float mass = volume * density;
    return mass;
}

// calculate kinetic energy of the system
float getSystemKE(float rad, float density, ChGranularSMC_API &apiSMC, int numSpheres){
    float sysKE = 0.0f;
    float sphere_KE;
    ChVector<float> angularVelo;
    ChVector<float> velo;
    float mass = getMass(rad, density);
    float inertia = 0.4f * mass * std::pow(rad,2);

    for (int i = 0; i < numSpheres; i++){
        angularVelo = apiSMC.getAngularVelo(i);
        velo = apiSMC.getVelo(i);
        sphere_KE = 0.5f * mass * velo.Length2() + 0.5f * inertia * angularVelo.Length2();
        sysKE = sysKE + sphere_KE;
    }
    return sysKE * KE_CGS_TO_SI;
}

// calucate total gravity of the system in SI unit
float getTotalGraivty(float grav_Z, float rad, float density, int numSpheres){
    return std::abs(grav_Z) * getMass(rad, density) * numSpheres * F_CGS_TO_SI;
}

int main(int argc, char* argv[]) {

    // sphere parameters
    float sphere_radius = 0.5f;

    // box dim
    float box_X = 10.0f;
    float box_Y = 10.0f;
    float box_Z = 40.0f;
    

    // material based parameter
    float sphere_density = 7.8;
    float YoungsModulus = 2e8;
    float nu = 0.28f;
    float COR = 0.6f;
    float cohesion_ratio = 0.0f;
    float adhesion_ratio = 0.0f;

    // damping parameters
    float gamma_n = 1e5 * std::sqrt(sphere_radius);
    float gamma_t = 1e2;

    // friction coefficient
    float mu_s_s2s = 0.1;
    float mu_s_s2w = 0.2f;

    // rolling friction coefficient
    float mu_r = 0.0f;
    
    float sigma = (1 - std::pow(nu,2))/YoungsModulus;
    float Kn_s2s = 4.0f/(6.0f * sigma) * std::sqrt(sphere_radius/2.0f) * std::sqrt(sphere_radius);
    float Kn_s2w = 4.0f/(6.0f * sigma) * sphere_radius;

    // set gravity
    float grav_X = 0.0f;
    float grav_Y = 0.0f;
    float grav_Z = -980.0f;

    // time integrator
    float step_size = 3e-5;
    float time_end = 3.0f;

    // setup simulation gran_sys
    ChSystemGranularSMC gran_sys(sphere_radius, sphere_density,
                                 make_float3(box_X*1.2, box_Y*1.2, box_Z*1.2));

    ChGranularSMC_API apiSMC;

    checkStepSize(sphere_radius, sphere_density, Kn_s2s, step_size);
    apiSMC.setGranSystem(&gran_sys);

    // float box_X = 10.0f;
    // float box_Y = 10.0f;
    // float box_Z = 40.0f;


    // assign box position
    float bottom_plate_pos[3] = {0.0f, 0.0f, -box_Z/2.0f};
    float left_plate_pos[3]   = {-box_X/2.0f, 0.0f, 0.0f};
    float right_plate_pos[3]  = { box_X/2.0f, 0.0f, 0.0f};
    float front_plate_pos[3]  = {0.0f,  box_Y/2.0f, 0.0f};
    float back_plate_pos[3]   = {0.0f, -box_Y/2.0f, 0.0f};

    // assign box plate normal
    float bottom_plate_normal[3] = { 0.0f, 0.0f, 1.0f};
    float left_plate_normal[3]   = { 1.0f, 0.0f, 0.0f};
    float right_plate_normal[3]  = {-1.0f, 0.0f, 0.0f};
    float front_plate_normal[3]  = { 0.0f,-1.0f, 0.0f};
    float back_plate_normal[3]   = { 0.0f, 1.0f, 0.0f};

    // create box
    size_t bottom_plate_id = gran_sys.Create_BC_Plane(bottom_plate_pos, bottom_plate_normal, true);
    size_t left_plate_id   = gran_sys.Create_BC_Plane(left_plate_pos,   left_plate_normal,   true);
    size_t right_plate_id  = gran_sys.Create_BC_Plane(right_plate_pos,  right_plate_normal,  true);
    size_t front_plate_id  = gran_sys.Create_BC_Plane(front_plate_pos,  front_plate_normal,  true);
    size_t back_plate_id   = gran_sys.Create_BC_Plane(back_plate_pos,   back_plate_normal,   true);


    ChVector<float> hdims((float)(box_X / 2.0 - sphere_radius),
                          (float)(box_Y / 2.0 - sphere_radius),
                          (float)(box_Z / 2.0 - sphere_radius));

    ChVector<float> center(0.f, 0.f, 0.0f);

    // Fill box with bodies
    std::vector<ChVector<float>> initialPos =
        utils::PDLayerSampler_BOX<float>(center, hdims, 2.f * sphere_radius, 1.05f);

    int numSpheres = initialPos.size();

    apiSMC.setElemsPositions(initialPos);

    float psi_T = 32.0f;
    float psi_L = 256.0f;
    gran_sys.setPsiFactors(psi_T, psi_L);



    // normal force model
    gran_sys.set_K_n_SPH2SPH(Kn_s2s);
    gran_sys.set_K_n_SPH2WALL(Kn_s2w);
    gran_sys.set_Gamma_n_SPH2SPH(gamma_n);
    gran_sys.set_Gamma_n_SPH2WALL(gamma_n);

    std::cout << "normal force model: " << "kn = " << Kn_s2s << ", " << "gn = " << gamma_n; 
    
    //tangential force model
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_K_t_SPH2SPH(2.0f/7.0f * Kn_s2s);
    gran_sys.set_K_t_SPH2WALL(2.0f/7.0f * Kn_s2w);
    gran_sys.set_Gamma_t_SPH2SPH(gamma_t);
    gran_sys.set_Gamma_t_SPH2WALL(gamma_t);
    gran_sys.set_static_friction_coeff_SPH2SPH(mu_s_s2s);
    gran_sys.set_static_friction_coeff_SPH2WALL(mu_s_s2w);

    std::cout << "tangential force model: " << "kn = " << 2.0f/7.0f * Kn_s2s << ", " << "gn = " << gamma_t; 


    // rolling friction model
    gran_sys.set_rolling_mode(GRAN_ROLLING_MODE::SCHWARTZ);
    gran_sys.set_rolling_coeff_SPH2SPH(mu_r);
    gran_sys.set_rolling_coeff_SPH2WALL(mu_r);


    gran_sys.set_Cohesion_ratio(cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2W(adhesion_ratio);
    gran_sys.set_gravitational_acceleration(grav_X, grav_Y, grav_Z);

    

    // Set the position of the BD
    gran_sys.set_BD_Fixed(true);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::FORWARD_EULER);
    gran_sys.set_fixed_stepSize(step_size);


    gran_sys.initialize();
    
    float curr_time = 0;

    // initialize values that I want to keep track of
    float sysKE;
    float leftPlate_reaction_force[3];
    float rightPlate_reaction_force[3];
    float frontPlate_reaction_force[3];
    float backPlate_reaction_force[3];
    float bottomPlate_reaction_force[3];

    float target_fz;
    float total_fz;
    // print out stepsize
    float print_time_step = 1000.0f * step_size;

    string output_dir = "three_sphere_stacking_box";
    filesystem::create_directory(filesystem::path(output_dir));

    char filename[100];
    int currframe = 1;


    while (curr_time < time_end) {

        printf("t = %.4f", curr_time);
        gran_sys.advance_simulation(print_time_step);
        curr_time += print_time_step;

        sysKE = getSystemKE(sphere_radius, sphere_density, apiSMC, numSpheres);
        std::cout << ", system KE: " << sysKE;

        gran_sys.getBCReactionForces(left_plate_id,  leftPlate_reaction_force);
        gran_sys.getBCReactionForces(right_plate_id, rightPlate_reaction_force);
        gran_sys.getBCReactionForces(front_plate_id, frontPlate_reaction_force);
        gran_sys.getBCReactionForces(back_plate_id,  backPlate_reaction_force);

        gran_sys.getBCReactionForces(bottom_plate_id, bottomPlate_reaction_force);
        std::cout << ", bot: " << bottomPlate_reaction_force[2] * F_CGS_TO_SI;

        total_fz = (leftPlate_reaction_force[2] + rightPlate_reaction_force[2] + bottomPlate_reaction_force[2] + frontPlate_reaction_force[2] + backPlate_reaction_force[2]) * F_CGS_TO_SI;

        std::cout << ", total fz: " << total_fz;
        target_fz = getTotalGraivty(grav_Z, sphere_radius, sphere_density, numSpheres);
        std::cout << ", " << "target fz: " << target_fz;

        std::cout << "\n";
        
        // write position info
        sprintf(filename, "%s/step%06d", output_dir.c_str(), currframe);
        gran_sys.writeFile(std::string(filename));
        currframe ++;



    }

    return 0;
}