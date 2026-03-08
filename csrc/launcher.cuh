#pragma once

#include "kittens.cuh"

#include "util.cuh"

MAKE_WORKER(launcher, detail::TIMING_EVENT_SPECIAL_LAUNCHER_START, detail::TIMING_EVENT_SPECIAL_LAUNCHER_END, 1)