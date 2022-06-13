#pragma once
#ifndef PSDR_CAMERATRANSIENT_H
#define PSDR_CAMERATRANSIENT_H

#include "camera.h"

enum TemporalType{
    TRAPEZOID,
    TOFSIN,
};

class TemporalSensitivity{
public:
    TemporalSensitivity(){};
    //TemporalSensitivity(Float tres_sensor, Float tres_light);
    TemporalSensitivity(int duration, Float tres_sensor, Float tres_light);
    /*inline Float start_time() const {
        //return tstart;
        return -tres_light;
    };
    inline Float end_time() const {
        //return tend;
        return tres_sensor;
    };*/
    //inline Float energy() const {return peak;}; // the implementation will be changed for other sensitivity functions than triangle pulses.
    virtual Float eval(Float time, int idx_tbin) const;// {assert(false);};
    virtual FloatAD eval(FloatAD time, int idx_tbin) const;// {assert(false);}; // Eval T_0(t), which is the correlated temporal sensitivity for the 0-th (first) bin
    virtual void bin_range(Float time, int& idx_start, int& idx_end) const;// {assert(false);}; // v10
private:
    TemporalType type;
    /// For Trapezoid
    inline Float start_time() const {
        //return tstart;
        return -tres_light;
    };
    inline Float end_time() const {
        //return tend;
        return tres_sensor;
    };
    //Float tstart; // time in ns
    //Float tend; // time in ns
    Float peak;
    Float tres_sensor;
    Float tres_light;
    Float tres_smaller; // == min(tres_sensor, tres_light)
    Float tres_larger;  // == max(tres_sensor, tres_light)

    /// For ToFSin
    int channels; // number of channels
    Float period; // (ns)
    Float omega; // angular frequency 2pi/period (rad/ns)
};
/*
class TemporalSensitivityTrapezoid: public TemporalSensitivity{
public:
    TemporalSensitivityTrapezoid(Float tres_sensor, Float tres_light);
    virtual Float eval(Float time, int idx_tbin) const;
    virtual FloatAD eval(FloatAD time, int idx_tbin) const;
    virtual void bin_range(Float time, int& idx_start, int& idx_end) const;
private:
    inline Float start_time() const {
        //return tstart;
        return -tres_light;
    };
    inline Float end_time() const {
        //return tend;
        return tres_sensor;
    };

    //Float tstart; // time in ns
    //Float tend; // time in ns
    Float peak;
    Float tres_sensor;
    Float tres_light;
    Float tres_smaller; // == min(tres_sensor, tres_light)
    Float tres_larger;  // == max(tres_sensor, tres_light)
};

class TemporalSensitivitySinToF: public TemporalSensitivity{
public:
    TemporalSensitivitySinToF(int channels, Float period);//Float tres_sensor, Float tres_light);
    virtual Float eval(Float time, int idx_tbin) const;
    virtual FloatAD eval(FloatAD time, int idx_tbin) const;
    virtual void bin_range(Float time, int& idx_start, int& idx_end) const;
private:
    Float period; // (ns)
    Float omega; // angular frequency 2pi/period (rad/ns)
    int channels; // number of channels
};
 */

struct CameraTransient: Camera {
    CameraTransient(){}
    CameraTransient(
            int width,
            int height,
            ptr<float> cam_to_world,
            ptr<float> cam_to_ndc,
            float clip_near,
            int med_id,
            ptr<float> velocities = ptr<float>(nullptr),
            int duration = 0,           // v10: # of ToF channels for ToF setting
            Float tres = 0.0,           // resolution (ns) of each temporal bin.
            Float tstart = 0.0,
            Float tres_light = -1.0    // resolution (ns) of the light source. A negative argument means the same value as tres.
            );
    //~CameraTransient();

    inline bool valid_transient () const //{return (duration > 0) && (tres > 0.0);}; // v9
                                         {return duration > 0;}; // v10: tres may == 0.0 (ToF)
    inline Float eval_tsens(Float time, int idx_tbin) const {return temp_sens.eval(time-tstart, idx_tbin);};
    inline FloatAD eval_tsens(FloatAD time, int idx_tbin) const {return temp_sens.eval(time-tstart, idx_tbin);}; // evaluate temporal sensitivity
    void bin_range(Float time, int& idx_start, int& idx_end) const {temp_sens.bin_range(time-tstart, idx_start, idx_end);};
    inline void bin_range(FloatAD time, int& idx_start, int& idx_end) const {bin_range(time.val, idx_start, idx_end);};
    int clip_bin_index(int& idx_start, int& idx_end) const;

    int duration;   // number of time bins
    Float tstart;   // time when the camera start to receive
    //Float tend;     // time when the camera end to receive
    Float tres;     // how long time each temporal bin receives
    TemporalSensitivity temp_sens;
    //TemporalSensitivity* temp_sens = NULL; // temporal sensitivity
};

#endif //PSDR_CAMERATRANSIENT_H
