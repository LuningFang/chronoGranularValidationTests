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
// Authors: Ruochun Zhang, Luning Fang
// =============================================================================
// validation test: single sphere rolling on a plane modeled as a mesh
// =============================================================================


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

    // big domain dimension
    params.box_X = 100;
    params.box_Y = 100;
    params.box_Z = 8 * params.sphere_radius;


    // define rolling friction coefficient
    float rolling_friction_ceofficient = 0.1f;
    params.rolling_friction_coeffS2S = rolling_friction_ceofficient;
    params.rolling_friction_coeffS2W = rolling_friction_ceofficient;
    params.rolling_friction_coeffS2M = rolling_friction_ceofficient;

    float time_step = params.step_size;

    ChGranularChronoTriMeshAPI apiSMC_TriMesh(params.sphere_radius, params.sphere_density,
                                              make_float3(params.box_X, params.box_Y, params.box_Z));

    ChSystemGranularSMC_trimesh& gran_sys = apiSMC_TriMesh.getGranSystemSMC_TriMesh();

    gran_sys.set_BD_Fixed(true);

    // define ground position in z position
    float ground_pos_z = -3 * params.sphere_radius;

    // calcuate settled position to make sure sphere barely touching the mesh
	double mass = 4.0/3.0 * CH_C_PI * pow(params.sphere_radius,3) * params.sphere_density;
	double penetration = pow(mass * abs(params.grav_Z) / params.normalStiffS2S * std::sqrt(params.sphere_radius), 2.0/3.0);	
	double settled_pos = ground_pos_z + params.sphere_radius - penetration;


    // assign initial position and velocity
    float initialVelo = 0.5f;
    std::vector<ChVector<float>> body_points; 
    std::vector<ChVector<float>> point_vels;
    std::vector<ChVector<float>> point_ang_vels;
    ChVector<float> only_point(-3.0f, -4.0f, settled_pos);
    ChVector<float> only_point_vel(initialVelo, 0.0f, 0.0f);
    ChVector<float> only_point_ang_vel(0.f, 0.0f, 0.0f);
    body_points.push_back(only_point);
    point_vels.push_back(only_point_vel);
    point_ang_vels.push_back(only_point_ang_vel);
    apiSMC_TriMesh.setElemsPositions(body_points, point_vels, point_ang_vels);

    // set normal force model
    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
    gran_sys.set_K_n_SPH2MESH(params.normalStiffS2M);
    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);
    gran_sys.set_Gamma_n_SPH2MESH(params.normalDampS2M);
    // set tangential force model
    gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
    gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
    gran_sys.set_K_t_SPH2MESH(params.tangentStiffS2M);
    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
    gran_sys.set_Gamma_t_SPH2MESH(params.tangentDampS2M);
    gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
    gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);
    gran_sys.set_static_friction_coeff_SPH2MESH(params.static_friction_coeffS2M);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);


    // set cohesion and adhesion model
    gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2M(params.adhesion_ratio_s2m);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

	// set rolling friction model
	gran_sys.set_rolling_mode(GRAN_ROLLING_MODE::SCHWARTZ);
	gran_sys.set_rolling_coeff_SPH2SPH(params.rolling_friction_coeffS2S);
	gran_sys.set_rolling_coeff_SPH2WALL(params.rolling_friction_coeffS2W);
    gran_sys.set_rolling_coeff_SPH2MESH(params.rolling_friction_coeffS2M);

    // set time integrator
    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);

    std::string mesh_filename("data/one_facet.obj");
    std::vector<string> mesh_filenames(1, mesh_filename);

    std::vector<float3> mesh_translations(1, make_float3(0.f, 0.f, ground_pos_z));

    float length = 5;
    float width = 5;
    float thickness = 0.5;
    std::vector<ChMatrix33<float>> mesh_rotscales(1, ChMatrix33<float>(1.f));

    float plate_density = 80;//params.sphere_density / 100.f;
    float plate_mass = (float)length * width * thickness * plate_density ;
    std::vector<float> mesh_masses(1, plate_mass);

    apiSMC_TriMesh.load_meshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);

    // gran_sys.setOutputMode(params.write_mode);
    // gran_sys.setVerbose(params.verbose);
    // filesystem::create_directory(filesystem::path(params.output_dir));

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

    unsigned int out_steps = (unsigned int)(1.0 / (out_fps * params.step_size));

    int currframe = 0;
    unsigned int curr_step = 0;
    clock_t start = std::clock();
    
    
    gran_sys.enableMeshCollision();
    
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> omega;
    float meshForces[6];

    double t = 0;

    while ( t < (double)params.time_end) {
        
        t += params.step_size;
        gran_sys.advance_simulation(params.step_size);

        pos = apiSMC_TriMesh.getPosition(0);
        velo = apiSMC_TriMesh.getVelo(0);
        omega = apiSMC_TriMesh.getAngularVelo(0);
        gran_sys.collectGeneralizedForcesOnMeshSoup(meshForces);

        printf("%e, %e, %e, %e, %e, %e, %e, %e, %e\n", t, pos.x(), pos.z(), velo.x(), velo.z(), omega.y(), meshForces[0], meshForces[1], meshForces[2]);

        

    }
    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;

    return 0;
}
