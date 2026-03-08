#pragma once

#include "kittens.cuh"

#include "util.cuh"

MAKE_WORKER(loader, detail::TIMING_EVENT_SPECIAL_LOADER_START, detail::TIMING_EVENT_SPECIAL_LOADER_END, 1)