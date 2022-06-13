#pragma once
#ifndef CONFIG_H__
#define CONFIG_H__


#ifndef NDER
#define NDER 1
#endif

#include<string>
#include<iostream>

// #define SHAPE_COMPUTE_VTX_NORMAL
constexpr int nder = NDER;
constexpr double ShadowEpsilon = 1e-3f;
constexpr double AngleEpsilon = 1e-3f;
constexpr double EdgeEpsilon = 1e-8;

// #define INCLUDE_NULL_BOUNDARIES


#endif