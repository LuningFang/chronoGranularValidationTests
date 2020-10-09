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

#include "chrono/core/ChGlobal.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChForce.h"
#include "chrono/timestepper/ChTimestepper.h"


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

// calculate kinetic energy of the system
float getSystemKE(float rad, float density, ChGranularChronoTriMeshAPI &apiSMC, int numSpheres){
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

void calculateBoundaryForces(ChSystemGranularSMC &gran_sys, 
                             int numSpheres, 
                             float rad,
                             float kn,
                             float gn,
                             float mass,
                             float bottom_plate_position,
                             std::vector<ChVector<float>> &normalForces,
                             std::vector<int> &particlesInContact){
    float3 velo;
    float penetration;
    float force_multiplier;
    float3 contact_normal = make_float3(0.0f, 0.0f, 1.0f);

    for (int i = 0; i < numSpheres; i++){
        float3 pos  = gran_sys.getPosition(i);

        // check if it's in contact with the bottom boundary
        if (pos.z - rad < bottom_plate_position){
            penetration = std::abs(pos.z - rad - bottom_plate_position);
            force_multiplier = sqrt(penetration/rad);
            float3 Fn = kn * penetration * contact_normal;

            velo = gran_sys.getVelocity(i);
            float3 rel_vel = velo;

            float projection = Dot(rel_vel, contact_normal);

            // add damping
            Fn = Fn + -1. * gn * projection * contact_normal * mass;
            Fn = Fn * force_multiplier;
            normalForces.push_back(ChVector<float>(Fn.x, Fn.y, Fn.z));
            particlesInContact.push_back(i);
        }

    }
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
    float sphere_radius = 1.0f;

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
    float time_end = 10;

    // setup simulation gran_sys
    ChGranularChronoTriMeshAPI apiSMC_TriMesh(sphere_radius, sphere_density, 
                                        make_float3(box_X, box_Y, box_Z));

    ChSystemGranularSMC_trimesh& gran_sys = apiSMC_TriMesh.getGranSystemSMC_TriMesh();

    std::vector<ChVector<float>> initialPos = initializePositions(x_dim_num, z_dim_num, sphere_radius);

    int numSpheres = initialPos.size();

    std::cout << "number of spheres: " << numSpheres;

    std::string mesh_filename("data/balldrop/sphere.obj");
    std::vector<std::string> mesh_filenames(1, mesh_filename);
    float ball_radius = sphere_radius;
    ChVector<float> ball_initial_pos(0.0f, 0.0f, initialPos.at(numSpheres-1).z()+2*sphere_radius);
    std::vector<float3> mesh_translations(1, make_float3(0.f, 0.f, 0.f));
    std::vector<ChMatrix33<float>> mesh_rotscales(1, ChMatrix33<float>(ball_radius));
    float ball_mass = 1.0f * sphere_mass;
    std::vector<float> mesh_masses(1, ball_mass);

    apiSMC_TriMesh.load_meshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);


    apiSMC_TriMesh.setElemsPositions(initialPos);
    unsigned int nSoupFamilies = gran_sys.getNumTriangleFamilies();

    double* meshPosRot = new double[7 * nSoupFamilies];
    float* meshVel = new float[6 * nSoupFamilies]();

    // create rigid_body simulation part
    ChSystemSMC sys_ball;
    sys_ball.SetContactForceModel(ChSystemSMC::ContactForceModel::Hertz);
    sys_ball.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT);
    sys_ball.Set_G_acc(ChVector<>(0, 0, -980));
    double inertia = 2.0 / 5.0 * ball_mass * sphere_radius * sphere_radius;

    std::shared_ptr<ChBody> ball_body(sys_ball.NewBody());
    ball_body->SetMass(ball_mass);
    ball_body->SetInertiaXX(ChVector<>(inertia, inertia, inertia));
    ball_body->SetPos(ball_initial_pos);
    sys_ball.AddBody(ball_body);



    float psi_T = 32.0f;
    float psi_L = 256.0f;
    gran_sys.setPsiFactors(psi_T, psi_L);


    // normal force model
    gran_sys.set_K_n_SPH2SPH(kn);
    gran_sys.set_K_n_SPH2WALL(2*kn);
    gran_sys.set_K_n_SPH2MESH(kn);
    gran_sys.set_Gamma_n_SPH2SPH(gamma_n);
    gran_sys.set_Gamma_n_SPH2WALL(gamma_n);
    gran_sys.set_Gamma_n_SPH2MESH(gamma_n);
    
    //tangential force model
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_K_t_SPH2SPH(2.0f/7.0f * kn);
    gran_sys.set_K_t_SPH2WALL(2.0f/7.0f * kn);
    gran_sys.set_K_t_SPH2MESH(2.0f/7.0f * kn);
    gran_sys.set_Gamma_t_SPH2SPH(gamma_t);
    gran_sys.set_Gamma_t_SPH2WALL(gamma_t);
    gran_sys.set_Gamma_t_SPH2MESH(gamma_t);

    gran_sys.set_static_friction_coeff_SPH2SPH(mu_s_s2s);
    gran_sys.set_static_friction_coeff_SPH2WALL(mu_s_s2w);
    gran_sys.set_static_friction_coeff_SPH2MESH(mu_s_s2s);


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
    int print_per_step = 5000;

    std::string output_dir = "wave_propagation_mesh_out";
    filesystem::create_directory(filesystem::path(output_dir));

    char filename[100];
    int currframe = 1;
    clock_t start = std::clock();
    int curr_step = 0;

    std::cout << "time, KE" << std::endl;
	while (curr_time < time_end) {
        auto ball_pos = ball_body->GetPos();
        auto ball_rot = ball_body->GetRot();

        auto ball_vel = ball_body->GetPos_dt();
        auto ball_ang_vel = ball_body->GetWvel_loc();
        ball_ang_vel = ball_body->GetRot().GetInverse().Rotate(ball_ang_vel);

        meshPosRot[0] = ball_pos.x();
        meshPosRot[1] = ball_pos.y();
        meshPosRot[2] = ball_pos.z();
        meshPosRot[3] = ball_rot[0];
        meshPosRot[4] = ball_rot[1];
        meshPosRot[5] = ball_rot[2];
        meshPosRot[6] = ball_rot[3];

        meshVel[0] = (float)ball_vel.x();
        meshVel[1] = (float)ball_vel.y();
        meshVel[2] = (float)ball_vel.z();
        meshVel[3] = (float)ball_ang_vel.x();
        meshVel[4] = (float)ball_ang_vel.y();
        meshVel[5] = (float)ball_ang_vel.z();

        gran_sys.meshSoup_applyRigidBodyMotion(meshPosRot, meshVel);

        gran_sys.advance_simulation(step_size);
        sys_ball.DoStepDynamics(step_size);

        float ball_force[6];
        gran_sys.collectGeneralizedForcesOnMeshSoup(ball_force);

        ball_body->Empty_forces_accumulators();
        ball_body->Accumulate_force(ChVector<>(ball_force[0], ball_force[1], ball_force[2]), ball_pos, false);
        ball_body->Accumulate_torque(ChVector<>(ball_force[3], ball_force[4], ball_force[5]), false);

        curr_time += step_size;
        curr_step = curr_step + 1;

        if (curr_step % print_per_step == 0){
            float KE = getSystemKE(sphere_radius, sphere_density, apiSMC_TriMesh, numSpheres);
            printf("t = %.4f", curr_time);

            std::cout << ", system KE: " << KE << " J";
            std::cout << "\n";

        }

    }

//        write position info
    sprintf(filename, "%s/settlingPosition", output_dir.c_str());
    gran_sys.writeFile(std::string(filename));


    std::vector<ChVector<float>> normalForces;
    std::vector<int> particlesInContact;
    calculateBoundaryForces(gran_sys, numSpheres, sphere_radius, 2*kn, gamma_n, sphere_mass, -box_Z/2.0f, normalForces, particlesInContact);

    printf("bottom layer reaction force: x, reaction force\n");
    for (int i = 0; i < normalForces.size(); i ++){
        printf("%e, %e\n", gran_sys.getPosition(particlesInContact.at(i)).x, normalForces.at(i).z());
    }

	clock_t end_time = std::clock();
	double computation_time = ((double)(end_time - start)) / CLOCKS_PER_SEC;
	std::cout << "Time: " << computation_time << " seconds" << std::endl;
    return 0;
}
