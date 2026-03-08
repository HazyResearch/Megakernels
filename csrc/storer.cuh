#pragma once

#include "kittens.cuh"
#include "util.cuh"

MAKE_WORKER(storer, detail::TIMING_EVENT_SPECIAL_STORER_START, detail::TIMING_EVENT_SPECIAL_STORER_END, 1)