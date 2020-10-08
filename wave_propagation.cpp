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
// Chrono::Granular test of spheres settling in a cylinder boundary 
// =============================================================================

#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
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
    std::cout << "usage: " + name + " <particle radius> " + " <gamma_n> " + " <gamma_t> "<< std::endl;
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

// initialize velocity, dimenstion of the slab: x_dim_num * radius by y_dim_num * radius
std::vector<ChVector<float>> initializePositions(int x_dim_num, int z_dim_num, float radius){
    std::vector<ChVector<float>> pos;
    float z = (-(float)z_dim_num/2.0f + 1.0f) * radius;
    float y = 0;
    float z_diff = std::sqrt(3.0f) * radius;
    float x;
    while (z <= ((float)z_dim_num/2.0f - 5.0f) * radius - z_diff){
        x = (-(float)x_dim_num/2.0f + 1.0f) * radius;
        while (x <= ((float)x_dim_num/2.0f - 1.0f) * radius){
            ChVector<float> position(x, y, z);
            pos.push_back(position);
            x = x + 2 * radius;
        }
        x = (-(float)x_dim_num/2.0f + 2.0f) * radius;
        z = z + z_diff;
        while (x <= ((float)x_dim_num/2.0f - 1.0f) * radius){
            ChVector<float> position(x, y, z);
            pos.push_back(position);
            x = x + 2 * radius;
        }
        z = z + z_diff;
    }
    return pos;

}

int main(int argc, char* argv[]) {

    // sphere parameters
    float sphere_radius = 1;

    int x_dim_num = 120;
    int z_dim_num = 30;

    // box dim
    float box_X = x_dim_num * sphere_radius;
    float box_Y = box_X;
    float box_Z = z_dim_num * sphere_radius;
    

    // material based parameter
    float sphere_density = 7.8;
    float sphere_volume = 4.0f/3.0f * CH_C_PI * std::pow(sphere_radius, 3);
    float sphere_mass = sphere_density * sphere_volume;

    float gravity = 980;
    float kn = 3000.0f * sphere_mass * gravity/sphere_radius;
    float kt = 0.0f;
    // damping parameters
    float gamma_n = 10000;
    float gamma_t = 0.0f;


    // friction coefficient
    float mu_s_s2s = 0.0f;
    float mu_s_s2w = 0.0f;

    // set gravity
    float grav_X = 0.0f;
    float grav_Y = 0.0f;
    float grav_Z = -gravity;

    // time integrator
    float step_size = 1e-5;
    float time_end = 15;

    // setup simulation gran_sys
    ChSystemGranularSMC gran_sys(sphere_radius, sphere_density,
                                 make_float3(box_X, box_Y, box_Z));

    ChGranularSMC_API apiSMC;
    apiSMC.setGranSystem(&gran_sys);

    std::vector<ChVector<float>> initialPos = initializePositions(x_dim_num, z_dim_num, sphere_radius);

    int numSpheres = initialPos.size();

    initialPos.push_back(ChVector<float>(0.0f, 0.0f, initialPos.at(numSpheres-1).z()+2*sphere_radius));

    std::cout << "number of spheres: " << numSpheres;

    apiSMC.setElemsPositions(initialPos);

    float psi_T = 32.0f;
    float psi_L = 256.0f;
    gran_sys.setPsiFactors(psi_T, psi_L);


    // normal force model
    gran_sys.set_K_n_SPH2SPH(kn);
    gran_sys.set_K_n_SPH2WALL(2*kn);
    gran_sys.set_Gamma_n_SPH2SPH(gamma_n);
    gran_sys.set_Gamma_n_SPH2WALL(gamma_n);
    
    //tangential force model
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_K_t_SPH2SPH(2.0f/7.0f * kn);
    gran_sys.set_K_t_SPH2WALL(2.0f/7.0f * kn);
    gran_sys.set_Gamma_t_SPH2SPH(gamma_t);
    gran_sys.set_Gamma_t_SPH2WALL(gamma_t);
    gran_sys.set_static_friction_coeff_SPH2SPH(mu_s_s2s);
    gran_sys.set_static_friction_coeff_SPH2WALL(mu_s_s2w);

    gran_sys.set_gravitational_acceleration(grav_X, grav_Y, grav_Z);

    

    // Set the position of the BD
    gran_sys.set_BD_Fixed(true);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::FORWARD_EULER);
    gran_sys.set_fixed_stepSize(step_size);


    gran_sys.initialize();
    
    float curr_time = 0;

    // initialize values that I want to keep track of
    float sysKE;
    // print out stepsize
    float print_time_step = 5000.0f * step_size;

    std::string output_dir = "wave_propagation_out";
    filesystem::create_directory(filesystem::path(output_dir));

    char filename[100];
    int currframe = 1;
    clock_t start = std::clock();

	while (curr_time < time_end) {
//        write position info
        sprintf(filename, "%s/step%06d", output_dir.c_str(), currframe);
        gran_sys.writeFile(std::string(filename));
        currframe ++;

        printf("t = %.4f", curr_time);
        gran_sys.advance_simulation(print_time_step);
        curr_time += print_time_step;

        sysKE = getSystemKE(sphere_radius, sphere_density, apiSMC, numSpheres);
        std::cout << ", system KE: " << sysKE << " J";

        // gran_sys.getBCReactionForces(cyl_id, cyl_reaction_force);
        // // std::cout << ", cyl wall: " << cyl_reaction_force[0] * F_CGS_TO_SI << ", " << cyl_reaction_force[1] * F_CGS_TO_SI << ", " << cyl_reaction_force[2] * F_CGS_TO_SI;

        // gran_sys.getBCReactionForces(plate_id, plane_reaction_force);
        // // std::cout << ", bottom plate: " << plane_reaction_force[0] * F_CGS_TO_SI << ", " << plane_reaction_force[1] * F_CGS_TO_SI << ", " << plane_reaction_force[2] * F_CGS_TO_SI;

        // total_fz = (cyl_reaction_force[2] + plane_reaction_force[2]) * F_CGS_TO_SI;

        // printf(", total fz: %e N", total_fz);
        // target_fz = getTotalGraivty(grav_Z, sphere_radius, sphere_density, numSpheres);
        // std::cout << ", " << "target fz: " << target_fz << " N";

        std::cout << "\n";
        



    }
	
	clock_t end_time = std::clock();
	double computation_time = ((double)(end_time - start)) / CLOCKS_PER_SEC;
	std::cout << "Time: " << computation_time << " seconds" << std::endl;
    return 0;
}
