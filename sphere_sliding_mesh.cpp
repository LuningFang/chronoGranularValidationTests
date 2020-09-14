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
//This demo simulate a plate intrude into a box area of granular materials with certain attack and intrusion angles

#include <iostream>
#include <vector>
#include <string>
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/core/ChGlobal.h"
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChForce.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/timestepper/ChTimestepper.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/physics/ChGranularTriMesh.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"

using namespace chrono;
using namespace chrono::granular;

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file>" << std::endl;
}

int main(int argc, char* argv[]) {

    sim_param_holder params;
    if (argc != 2 || ParseJSON(argv[1], params) == false) {
        ShowUsage(argv[0]);
        return 1;
    }

    float iteration_step = params.step_size;

    ChGranularChronoTriMeshAPI apiSMC_TriMesh(params.sphere_radius, params.sphere_density,
                                              make_float3(params.box_X, params.box_Y, params.box_Z));

    ChSystemGranularSMC_trimesh& gran_sys = apiSMC_TriMesh.getGranSystemSMC_TriMesh();

    std::vector<ChVector<float>> body_points; 
    std::vector<ChVector<float>> point_vels;
    std::vector<ChVector<float>> point_ang_vels;
    ChVector<float> only_point(1.f, -1.f, 1.0f*params.sphere_radius-0.1f);
    ChVector<float> only_point_vel(1.f, 0.0f, 0.0f);
    ChVector<float> only_point_ang_vel(0.f, 0.0f, 0.0f);
    body_points.push_back(only_point);
    point_vels.push_back(only_point_vel);
    point_ang_vels.push_back(only_point_ang_vel);
    //apiSMC_TriMesh.setElemsPositions(body_points);
    std::vector<float3> locationFloat3;
    std::vector<float3> velFloat3;
    std::vector<float3> angvelFloat3;
    convertChVector2Float3Vec(body_points, locationFloat3);
    convertChVector2Float3Vec(point_vels, velFloat3);
    convertChVector2Float3Vec(point_ang_vels, angvelFloat3);
    gran_sys.setParticlePositions(locationFloat3, velFloat3, angvelFloat3);
    //gran_sys.setParticlePositions(locationFloat3, velFloat3);

    gran_sys.set_BD_Fixed(true);

    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
    gran_sys.set_K_n_SPH2MESH(params.normalStiffS2M);

    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);
    gran_sys.set_Gamma_n_SPH2MESH(params.normalDampS2M);

    gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
    gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
    gran_sys.set_K_t_SPH2MESH(params.tangentStiffS2M);

    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
    gran_sys.set_Gamma_t_SPH2MESH(params.tangentDampS2M);

    gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2M(params.adhesion_ratio_s2m);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
    gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
    gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);
    gran_sys.set_static_friction_coeff_SPH2MESH(params.static_friction_coeffS2M);

    gran_sys.setOutputFlags(GRAN_OUTPUT_FLAGS::ABSV | GRAN_OUTPUT_FLAGS::ANG_VEL_COMPONENTS | GRAN_OUTPUT_FLAGS::VEL_COMPONENTS);
    std::string mesh_filename("data/one_facet.obj");
    std::vector<string> mesh_filenames(1, mesh_filename);

    std::vector<float3> mesh_translations(1, make_float3(0.f, 0.f, -0.1f));

    float ball_radius = 20.f;
    float length = 5;
    float width = 5;
    float thickness = 0.5;
    std::vector<ChMatrix33<float>> mesh_rotscales(1, ChMatrix33<float>(1.f));

    float plate_density = 80;//params.sphere_density / 100.f;
    float plate_mass = (float)length * width * thickness * plate_density ;
    std::vector<float> mesh_masses(1, plate_mass);

    apiSMC_TriMesh.load_meshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);

    gran_sys.setOutputMode(params.write_mode);
    gran_sys.setVerbose(params.verbose);
    filesystem::create_directory(filesystem::path(params.output_dir));

    unsigned int nSoupFamilies = gran_sys.getNumTriangleFamilies();
    std::cout << nSoupFamilies << " soup families" << std::endl;
    double* meshPosRot = new double[7 * nSoupFamilies];
    float* meshVel = new float[6 * nSoupFamilies]();

    //float plane_pos[3] = {0, 0, 0};
    //float plane_normal[3] = {0, 0, 1};
    //size_t plane_bc_id = gran_sys.Create_BC_Plane(plane_pos, plane_normal, false);

    gran_sys.initialize();

    unsigned int out_fps = 50;
    std::cout << "Rendering at " << out_fps << "FPS" << std::endl;

    unsigned int out_steps = (unsigned int)(1.0 / (out_fps * iteration_step));

    int currframe = 0;
    unsigned int curr_step = 0;
    clock_t start = std::clock();
    
    int counter = 0;
    
    gran_sys.enableMeshCollision();
    //gran_sys.disableMeshCollision();
    
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> omega;

    for (double t = 0; t < (double)params.time_end; t += iteration_step, curr_step++) {

        gran_sys.advance_simulation(iteration_step);

        if (curr_step % out_steps == 0) {
            std::cout << "Rendering frame " << currframe << std::endl;
                char filename[100];
                sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe++);
                gran_sys.writeFile(std::string(filename));
                gran_sys.write_meshes(std::string(filename));
		    
        }

        pos = apiSMC_TriMesh.getPosition(0);
        printf("particle position: %f, %f, %f\n", pos.x(), pos.y(), pos.z());

        counter++;
    }
    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;

    return 0;
}
