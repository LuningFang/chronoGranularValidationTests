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
// John's test
// =============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "chrono/core/ChGlobal.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_gpu/utils/ChGpuJsonParser.h"
#include "chrono_gpu/ChGpuData.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::gpu;

#define FAM_ENTRIES_POS 7
#define FAM_ENTRIES_VEL 6
#define FAM_ENTRIES_FORCE 6

// Normal stress values for four tests (g/cm)
double normal_stresses[] = {3.1e4, 6.4e4, 12.5e4, 24.2e4};
double plate_mass;

double shear_displacement = 1;         // X displacement at which the test ends
double shear_velocity = 0.1;           // shear velocity 1mm/s

double box_xy = 12;                      // 12 cm by 12 cm box
double box_r = box_xy / 2;

// TODO tune these values
double time_settle = 0.4;
double time_compress = 1;
double time_shear = shear_displacement / shear_velocity;

// Indices of each object
const size_t bottom_i = 0;
const size_t top_i = 1;
const size_t plate_i = 2;

double fill_top;
double fill_bottom = -3.0f;

double box_X = 15;
double box_Y = 15;
double box_Z = 20.0f;

double step_size = 1e-5;
double sphere_radius = 0.3;
double sphere_density = 2.55;

double grav_X = 0.0f;
double grav_Y = 0.0f;
double grav_Z = -981.0f;

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <normal_stress_index>" << std::endl;
}

double getVoidRatio(double top_plate_pos, int nb, double sphere_radius){
    double vol_sphere = 4.0f/3.0f * chrono::CH_C_PI * std::pow(sphere_radius, 3) * nb;
    double vol_box = (top_plate_pos - fill_bottom) * box_xy * box_xy;

    double eta = vol_sphere/vol_box;
    double voidRatio = (1.0f - eta)/eta;
    return voidRatio;
}

double CalcKE(ChSystemGpuMesh& gpu_sys, double sphere_radius, double sphere_density, int nb){
    double vol_sphere = 4.0f/3.0f * chrono::CH_C_PI * std::pow(sphere_radius, 3);
    double mass_sphere = vol_sphere * sphere_density;
    double inertia_sphere = 0.4f * mass_sphere * sphere_radius * sphere_radius;
    ChVector<float> ang_velo;
    ChVector<float> lin_velo;
    double KE = 0;
    for (int i = 0; i < nb; i++){
        lin_velo  = gpu_sys.GetParticleVelocity(i);
        ang_velo = gpu_sys.GetParticleAngVelocity(i);
        KE = 0.5f * mass_sphere * lin_velo.Length2() + 0.5f * inertia_sphere * ang_velo.Length2();
    }
    return KE;
}

void SetupGranSystem(ChSystemGpuMesh& gpu_sys) {

        double cor_p = 0.87;
        double cor_w = 0.5;
        double youngs_modulus = 4e8; // 70Mpa = 7e7Pa = 7e8 g/(cms^2)
        double poisson_ratio = 0.22;
        double mu_s2s = 0.18;
        double mu_s2w = 0.40;  // what's this?
            
        gpu_sys.UseMaterialBasedModel(true);

        gpu_sys.SetYoungModulus_SPH(youngs_modulus);
        gpu_sys.SetYoungModulus_WALL(youngs_modulus);            
        gpu_sys.SetYoungModulus_MESH(youngs_modulus);

        gpu_sys.SetRestitution_SPH(cor_p);
        gpu_sys.SetRestitution_WALL(cor_w);
        gpu_sys.SetRestitution_MESH(cor_w);

        gpu_sys.SetPoissonRatio_SPH(poisson_ratio);
        gpu_sys.SetPoissonRatio_WALL(poisson_ratio);
        gpu_sys.SetPoissonRatio_MESH(poisson_ratio);
        gpu_sys.SetGravitationalAcceleration(ChVector<float>(grav_X, grav_Y, grav_Z));

        gpu_sys.SetFrictionMode(chrono::gpu::CHGPU_FRICTION_MODE::MULTI_STEP);
        gpu_sys.SetStaticFrictionCoeff_SPH2SPH(mu_s2s);
        gpu_sys.SetStaticFrictionCoeff_SPH2WALL(mu_s2w);
        gpu_sys.SetStaticFrictionCoeff_SPH2MESH(mu_s2w);

        gpu_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::CSV);
        gpu_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
        gpu_sys.SetFixedStepSize(step_size);
        gpu_sys.SetBDFixed(true);


        // start outside BD by 10 cm
        ChVector<float> plane_pos(0.0f, 0.0f, fill_bottom);
        ChVector<float> plane_normal(0, 0, 1.0f);

        size_t plane_bc_id = gpu_sys.CreateBCPlane(plane_pos, plane_normal, false);


        double spacing = 2.001 * sphere_radius;

        std::vector<ChVector<float>> body_points;

        utils::PDSampler<float> sampler(spacing);
        fill_top = box_Z / 2 - spacing;  // TODO tune to roughly make a cube of material (6cm tall)

        ChVector<> hdims(box_r - sphere_radius, box_r - sphere_radius, 0);
        int counter = 0;
        for (double z = fill_bottom + spacing; z < fill_top; z += spacing) {
            ChVector<> center(0, 0, z);
            auto points = sampler.SampleBox(center, hdims);
            body_points.insert(body_points.end(), points.begin(), points.end());
            counter = counter + points.size();
            if (counter > 5000){
                break;
            }
        }   

        std::cout << "Created " << body_points.size() << " spheres" << std::endl;

        gpu_sys.SetParticles(body_points);

        // Mesh values
        std::vector<string> mesh_filenames;
        // TODO dull the corners and fix nans
        mesh_filenames.push_back(std::string(gpu::GetDataFile("shear_bottom.obj")));
        mesh_filenames.push_back(std::string(gpu::GetDataFile("shear_top.obj")));
        mesh_filenames.push_back(std::string(gpu::GetDataFile("downward_square.obj")));

        ChMatrix33<float> scale(ChVector<float>(box_r, box_r, box_r));
        std::vector<ChMatrix33<float>> mesh_rotscales = {scale, scale, scale};
        std::vector<ChVector<float>> mesh_translations = {ChVector<float>(0, 0, 0), ChVector<float>(0, 0, 0),
                                                      ChVector<float>(0, 0, 0)};
        std::vector<float> mesh_masses = {1000, 1000, (float)plate_mass};

        gpu_sys.AddMeshes(mesh_filenames, mesh_translations, mesh_rotscales, mesh_masses);
}

void SetInitialMeshes(ChSystemGpuMesh& gpu_sys, const std::shared_ptr<ChBody> plate) {
    // initial positions and velocity
    ChVector<float> mesh_pos(0, 0, 0);
    ChQuaternion<float> mesh_rot(1, 0, 0, 0);
    ChVector<float> mesh_lin_vel(0, 0, 0);
    ChVector<float> mesh_ang_vel(0, 0 , 0);
    
    // Bottom bin
    gpu_sys.ApplyMeshMotion(bottom_i, mesh_pos, mesh_rot, mesh_lin_vel, mesh_ang_vel);

    // Top bin
    gpu_sys.ApplyMeshMotion(top_i, mesh_pos, mesh_rot, mesh_lin_vel, mesh_ang_vel);

    // Plate
    ChVector<float> plate_pos(0, 0, box_Z/2.0f);
    gpu_sys.ApplyMeshMotion(plate_i, plate_pos, mesh_rot, mesh_lin_vel, mesh_ang_vel);
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        ShowUsage(argv[0]);
        return 1;
    }

    int normal_stress_id = std::atoi(argv[1]);

    ChSystemGpuMesh gran_sys(sphere_radius, sphere_density,
                            make_float3(box_X, box_Y, box_Z));

    std::string out_dir = GetChronoOutputPath() + "shear/";
    filesystem::create_directory(filesystem::path(out_dir));

    SetupGranSystem(gran_sys);
    gran_sys.Initialize();

    unsigned int numMeshes = gran_sys.GetNumMeshes();
    std::cout << numMeshes << " soup families" << std::endl;


    unsigned int currframe = 0;
    float out_fps = 100;
    float frame_step = 1.f / out_fps;  // Duration of a frame
    unsigned int out_steps = (unsigned int)(frame_step / step_size);    

    double m_time = 0;
    unsigned int step = 0;

    ChSystemSMC ch_sys;
    double grav_mag = std::sqrt(grav_X * grav_X + grav_Y * grav_Y + grav_Z * grav_Z);

    ch_sys.Set_G_acc(ChVector<>(grav_X, grav_Y, grav_Z));

    auto plate = std::make_shared<ChBody>();
    plate->SetBodyFixed(true);
    plate->SetPos(ChVector<>(0, 0, box_Z/2.0));  // Initially out of the way
    plate_mass = normal_stresses[normal_stress_id] * box_xy * box_xy / grav_mag;
    plate->SetMass(plate_mass);
    ch_sys.AddBody(plate);

    SetInitialMeshes(gran_sys, plate);

    int nb = gran_sys.GetNumParticles();

    std::cout << "Running settling..." << std::endl;
    for (; m_time < time_settle; m_time += step_size, step++) {
        if (step % out_steps == 0) {
            char filename[100];
            sprintf(filename, "%s/step%06d.csv", out_dir.c_str(), currframe++);
            gran_sys.WriteParticleFile(std::string(filename));
            gran_sys.WriteMeshes(std::string(filename));

            double sysKE = CalcKE(gran_sys, sphere_radius, sphere_density, nb);
            printf("%e, %e\n", m_time, sysKE);



        }
        gran_sys.AdvanceSimulation(step_size);



    }

    // Add a weighted top plate
    double plate_z = gran_sys.GetMaxParticleZ() + 2 * sphere_radius;
    std::cout << "Adding plate at "
              << "(0, 0, " << plate_z << ")" << std::endl;
    plate->SetPos(ChVector<>(0, 0, plate_z));
    plate->SetBodyFixed(false);

    float* forces = new float[numMeshes * FAM_ENTRIES_FORCE];

    // Compress the material under the weight of the plate
    std::cout << "Running compression..." << std::endl;
    m_time = 0;
    ChVector<float> plate_pos(0, 0, 0);
    ChQuaternion<float> plate_quat(1, 0, 0, 0);
    ChVector<float> plate_lin_velo(0, 0, 0);
    ChVector<float> plate_ang_velo(0, 0, 0);
    ChVector<> plate_force;
    ChVector<> plate_torque;

    ChVector<float> top_pos(0, 0, 0);
    ChQuaternion<float> top_quat(1, 0, 0, 0);
    ChVector<float> top_lin_velo(0, 0, 0);
    ChVector<float> top_rot_velo(0, 0, 0);

    ChVector<float> bottom_pos(0, 0, 0);
    ChQuaternion<float> bottom_quat(1, 0, 0, 0);
    ChVector<float> bottom_lin_velo(0, 0, 0);
    ChVector<float> bottom_rot_velo(0, 0, 0);


    for (; m_time < time_compress; m_time += step_size, step++) {
        // Update Plate
        plate_pos.z() = plate->GetPos().z();
        plate_lin_velo.z() = plate->GetPos_dt().z();

        gran_sys.ApplyMeshMotion(plate_i, plate_pos, plate_quat, plate_lin_velo, plate_ang_velo);

        if (step % out_steps == 0) {
            char filename[100];
            sprintf(filename, "%s/step%06d.csv", out_dir.c_str(), currframe++);
            gran_sys.WriteParticleFile(std::string(filename));
            gran_sys.WriteMeshes(std::string(filename));
            double voidRatio = getVoidRatio(plate_pos.z(), nb, sphere_radius);
            double sysKE = CalcKE(gran_sys, sphere_radius, sphere_density,nb);
            printf("%e, %e, %e, %e\n", m_time, voidRatio, plate_pos.z(), sysKE);



        }

        ch_sys.DoStepDynamics(step_size);
        gran_sys.AdvanceSimulation(step_size);

        gran_sys.CollectMeshContactForces(plate_i, plate_force, plate_torque);
        plate->Empty_forces_accumulators();
        // set force in x and y direction to zero
        plate_force.x() = 0;
        plate_force.y() = 0;
        plate->Accumulate_force(plate_force, plate->GetPos(), false);


    }

    std::cout << std::endl << "Running shear test..." << std::endl;

    m_time = 0;
    for (; m_time < time_shear; step++, m_time += step_size) {
        double pos = m_time * shear_velocity;

        // Update Plate
        plate_pos.x() = pos;
        plate_pos.z() = plate->GetPos().z();
        plate_lin_velo.x() = shear_velocity;
        plate_lin_velo.z() = plate->GetPos_dt().z();
        gran_sys.ApplyMeshMotion(plate_i, plate_pos, plate_quat, plate_lin_velo, plate_ang_velo);
        
        // Update top bin
        top_pos.x() = pos;
        top_lin_velo.x() = shear_velocity;
        gran_sys.ApplyMeshMotion(top_i, top_pos, top_quat, top_lin_velo, top_rot_velo);        

        gran_sys.AdvanceSimulation(step_size);
        ch_sys.DoStepDynamics(step_size);

        gran_sys.CollectMeshContactForces(plate_i, plate_force, plate_torque);
        ChVector<> top_bin_force;
        ChVector<> top_bin_torque;
        gran_sys.CollectMeshContactForces(top_i, top_bin_force, top_bin_torque);
        double shear_force = plate_force.x() + top_bin_force.x();

        plate->Empty_forces_accumulators();
        plate->Accumulate_force(ChVector<>(0, 0, plate_force.z()), plate->GetPos(), false);

        // shear_force = fm_lowpass5.Filter(shear_force);

        // Output displacement and force
        if (step % out_steps == 0) {
            char filename[100];
            sprintf(filename, "%s/step%06d.csv", out_dir.c_str(), currframe++);
            gran_sys.WriteParticleFile(std::string(filename));
            gran_sys.WriteMeshes(std::string(filename));

            double shear_area = box_xy * (box_xy - m_time * shear_velocity * 2);
            double normal_stress = (plate_mass * grav_mag) / shear_area;
            double shear_stress = shear_force / shear_area;
            printf("%e, %e, %e, %e\n", m_time, pos*10.0f, normal_stress/1e4, shear_stress/1e4);
            // std::cout << std::setprecision(4) << "Time: " << m_time << std::endl;
            // std::cout << std::setprecision(4) << "\tShear displacement: " << pos << std::endl;
            // std::cout << std::setprecision(4) << "\tNormal stress: " << normal_stress << std::endl;
            // std::cout << std::setprecision(4) << "\tShear stress: " << shear_stress << std::endl;
            // std::cout << std::setprecision(4) << "\tShear stress / Normal stress: " << shear_stress / normal_stress
            //           << std::endl;
        }
    }

    return 0;
}