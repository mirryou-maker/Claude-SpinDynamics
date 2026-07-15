// Shared includes for the split _micromag binding translation units.
// Generated from the former monolithic bindings.cpp (1,824 lines) — each
// bind_*.cpp registers one domain so incremental builds only recompile the
// part that changed.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "micromag/types.hpp"
#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/vtk_writer.hpp"
#include "micromag/material.hpp"
#include "micromag/material_field.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/exchange.hpp"
#include "micromag/demag.hpp"
#include "micromag/demag_periodic.hpp"
#include "micromag/rkky.hpp"
#include "micromag/zeeman_spatial.hpp"
#include "micromag/geom_mask.hpp"
#include "micromag/mfm.hpp"
#include "micromag/integrator.hpp"
#include "micromag/thermal_field.hpp"
#include "micromag/spin_torque.hpp"
#include "micromag/dmi.hpp"
#include "micromag/solver.hpp"
#include "micromag/ovf_io.hpp"
#include "micromag/cubic_anisotropy.hpp"
#include "micromag/surface_anisotropy.hpp"
#include "micromag/magnetoelastic.hpp"
#include "micromag/region_map.hpp"
#include "micromag/init_mag.hpp"
#include "micromag/topological_charge.hpp"
#include "micromag/skyrmion_tools.hpp"

#include "micromag/gpu_real.hpp"


namespace py = pybind11;
using namespace micromag;
