#pragma once

#include "kittens.cuh"
#include "util.cuh"

MAKE_WORKER(consumer, detail::TIMING_EVENT_SPECIAL_CONSUMER_START, detail::TIMING_EVENT_SPECIAL_CONSUMER_END, config::NUM_CONSUMER_WARPS)