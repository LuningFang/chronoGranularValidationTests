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
// Granular material settling in a square container to generate the bedding for 
// balldrop test, once settled positions are written
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
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
#include "ChGranularDemoUtils.hpp"

using namespace chrono;
using namespace chrono::granular;

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file>" << std::endl;
    std::cout << "OR " + name + " <json_file> "  + " <input_file> " <<std::endl;
}

double getMass(sim_param_holder& params){
    double rad = params.sphere_radius;
    double density = params.sphere_density;

    double volume = 4.0f/3.0f * CH_C_PI * std::pow(rad, 3);
    double mass = volume * density;
    return mass;
}

// calculate kinetic energy of the system
double getSystemKE(sim_param_holder &params, ChGranularSMC_API &apiSMC, int numSpheres){
    double sysKE = 0.0f;
    double sphere_KE;
    ChVector<float> angularVelo;
    ChVector<float> velo;
    double mass = getMass(params);
    double inertia = 0.4f * mass * std::pow(params.sphere_radius,2);

    for (int i = 0; i < numSpheres; i++){
        angularVelo = apiSMC.getAngularVelo(i);
        velo = apiSMC.getVelo(i);
        sphere_KE = 0.5f * mass * velo.Length2() + 0.5f * inertia * angularVelo.Length2();
        sysKE = sysKE + sphere_KE;
    }
    return sysKE;
}

int main(int argc, char* argv[]) {
    sim_param_holder params;
    bool useCheckpointInput = false;

    if ((argc != 2 && argc != 3) || ParseJSON(argv[1], params) == false){
        ShowUsage(argv[0]);
        return 1;
    }
    if (argc == 3 ) {
        useCheckpointInput = true;
    }

    float iteration_step = params.step_size;

    ChSystemGranularSMC gran_sys(params.sphere_radius, params.sphere_density,
                                 make_float3(params.box_X, params.box_Y, params.box_Z));

    ChGranularSMC_API apiSMC;
    apiSMC.setGranSystem(&gran_sys);


    double fill_bottom = -params.box_Z / 2.0;
    double fill_top = params.box_Z / 4.0;

    // declar particle positon vector
    std::vector<chrono::ChVector<float>> body_points; 
    if (useCheckpointInput == true){
        body_points = loadPositionCheckpoint<float>(argv[2]);
		std::cout << "reading position input success from " << argv[2]<<std::endl;
	}
    else
    {
        chrono::utils::PDSampler<float> sampler(2.1f * params.sphere_radius);

        ChVector<> hdims(params.box_X / 2 - 2*params.sphere_radius, params.box_Y / 2 - 2*params.sphere_radius, 0);
        ChVector<> center(0, 0, fill_bottom + 2.0 * params.sphere_radius);

        // Shift up for bottom of box
        while (center.z() < fill_top) {
            std::cout << "Create layer at " << center.z() << std::endl;
            auto points = sampler.SampleBox(center, hdims);
            body_points.insert(body_points.end(), points.begin(), points.end());
            center.z() += 2.1 * params.sphere_radius;
        }
    }
    
    int numSpheres = body_points.size();
    apiSMC.setElemsPositions(body_points);

    gran_sys.set_BD_Fixed(true);

    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);

    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);

    gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
    gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);

    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);

    gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
    gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
    gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);

    gran_sys.setOutputMode(params.write_mode);
    gran_sys.setVerbose(params.verbose);
    filesystem::create_directory(filesystem::path(params.output_dir));

    gran_sys.initialize();

    unsigned int out_fps = 50;
    std::cout << "Rendering at " << out_fps << "FPS" << std::endl;

    unsigned int out_steps = (unsigned int)(1.0 / (out_fps * iteration_step));

    int currframe = 0;
    unsigned int curr_step = 0;

    clock_t start = std::clock();
    char filename[100];

    for (double t = 0; t < (double)params.time_end; t += iteration_step, curr_step++) {

        gran_sys.advance_simulation(iteration_step);

        double KE;
        if (curr_step % out_steps == 0) {
            std::cout << "Rendering frame " << currframe;
            sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe++);
            gran_sys.writeFile(std::string(filename));
            KE = getSystemKE(params, apiSMC, numSpheres);

            std::cout << ", time = " << t << ", KE = " << KE * 1E-5 << " J, max z: " << gran_sys.get_max_z() << " cm" << std::endl;
		
		}
    }

    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;
    sprintf(filename, "%s/settled", params.output_dir.c_str());
    gran_sys.writeFile(std::string(filename));

    return 0;
}
