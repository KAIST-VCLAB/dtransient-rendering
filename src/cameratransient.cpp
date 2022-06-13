
#include "cameratransient.h"

TemporalSensitivity::TemporalSensitivity(int duration, Float tres_sensor, Float tres_light){
    if(tres_sensor > 0.0){
        /// Trapezoid
        type = TRAPEZOID;
        this->tres_sensor = tres_sensor;
        this->tres_light = tres_light;
        tres_smaller = std::min(tres_sensor, tres_light);
        tres_larger = std::max(tres_sensor, tres_light);
        peak = 1.0/tres_light; // unit energy normalization which makes the sum of each transient frame same as the steady-state one.
    } else{
        /// ToFSin
        type = TOFSIN;
        channels = duration;
        period = -tres_sensor;
        omega = 2*M_PI/period;
    }
}

Float TemporalSensitivity::eval(Float time, int idx_tbin) const
{
    switch(type){
        case TRAPEZOID: {
            time -= idx_tbin * tres_sensor;
            Float ret;
            if (time <= start_time() || time >= end_time())
                ret = Float(0.0);
            else if (time <= start_time() + tres_smaller)
                ret = time - start_time();
            else if (time >= end_time() - tres_smaller)
                ret = end_time() - time;
            else // start_time() + tres_smaller < time < end_time() - tres_smaller
                ret = Float(tres_smaller);

            return ret * peak;
        } case TOFSIN: {
            Float phase = time * omega;
            phase += 2 * M_PI * idx_tbin / channels;
            return sin(phase)+1;
        }
    }
}

FloatAD TemporalSensitivity::eval(FloatAD time, int idx_tbin) const
{
    switch(type){
        case TRAPEZOID: {
            time -= idx_tbin * tres_sensor;
            FloatAD ret;
            if (time.val <= start_time() || time.val >= end_time())
                ret = FloatAD(0.0, 0.0);
            else if (time.val <= start_time() + tres_smaller)
                ret = time - start_time();
            else if (time.val >= end_time() - tres_smaller)
                ret = end_time() - time;
            else // start_time() + tres_smaller < time.val < end_time() - tres_smaller
                ret = FloatAD(tres_smaller, 0.0);

            return ret * peak;
        } case TOFSIN: {
            FloatAD phase = time * omega;
            phase += 2 * M_PI * idx_tbin / channels;
            return phase.sin()+1;
        }
    }

}

void TemporalSensitivity::bin_range(Float time, int& idx_start, int& idx_end) const{
    switch(type){
        case TRAPEZOID:
            // Want to satisfy: temp_sens.start_time() < time-tres*idx_tbin < temp_sens.end_time()
            //             <==> (time - end_time) / tres < idx_tbin < (time - start_time) / tres
            idx_start = std::ceil((time - end_time()) / tres_sensor);
            idx_end = std::floor((time - start_time()) / tres_sensor);
            assert(idx_start <= idx_end);
            break;
        case TOFSIN:
            idx_start = 0;
            idx_end = channels-1;
            break;
    }
}

/////////////////////////////////////////////////////
/// TemporalSensitivityTrapezoid
/////////////////////////////////////////////////////
/*
//TemporalSensitivity::TemporalSensitivity(Float tstart, Float tend, Float peak): tstart(tstart), tend(tend), peak(peak) {}
TemporalSensitivityTrapezoid::TemporalSensitivityTrapezoid(Float tres_sensor, Float tres_light): tres_sensor(tres_sensor), tres_light(tres_light) {
    tres_smaller = std::min(tres_sensor, tres_light);
    tres_larger = std::max(tres_sensor, tres_light);
    peak = 1.0/tres_light; // unit energy normalization which makes the sum of each transient frame same as the steady-state one.
}

Float TemporalSensitivityTrapezoid::eval(Float time, int idx_tbin) const
{
    time -= idx_tbin*tres_sensor;
    Float ret;
    if (time <= start_time() || time >= end_time())
        ret =  Float(0.0);
    else if (time <= start_time() + tres_smaller)
        ret = time - start_time();
    else if (time >= end_time() - tres_smaller)
        ret = end_time() - time;
    else // start_time() + tres_smaller < time < end_time() - tres_smaller
        ret = Float(tres_smaller);

    return ret * peak;
}

FloatAD TemporalSensitivityTrapezoid::eval(FloatAD time, int idx_tbin) const
{
    /*if (time < tstart || time > tend)
        return FloatAD(0.0, 0.0);
    else
        return FloatAD(peak) - (time - (tstart+tend)/2).abs();* /
    time -= idx_tbin*tres_sensor;
    FloatAD ret;
    if (time.val <= start_time() || time.val >= end_time())
        ret =  FloatAD(0.0, 0.0);
    else if (time.val <= start_time() + tres_smaller)
        ret = time - start_time();
    else if (time.val >= end_time() - tres_smaller)
        ret = end_time() - time;
    else // start_time() + tres_smaller < time.val < end_time() - tres_smaller
        ret = FloatAD(tres_smaller, 0.0);

    return ret * peak;
}

void TemporalSensitivityTrapezoid::bin_range(Float time, int& idx_start, int& idx_end) const{
    // Want to satisfy: temp_sens.start_time() < time-tres*idx_tbin < temp_sens.end_time()
    //             <==> (time - end_time) / tres < idx_tbin < (time - start_time) / tres
    idx_start = std::ceil((time - end_time()) / tres_sensor);
    idx_end = std::floor((time - start_time()) / tres_sensor);
    assert(idx_start <= idx_end);
}
*/


/////////////////////////////////////////////////////
/// TemporalSensitivitySinToF
/////////////////////////////////////////////////////
/*
TemporalSensitivitySinToF::TemporalSensitivitySinToF(int channels, Float period): channels(channels), period(period){
    omega = 2*M_PI/period;
}

Float TemporalSensitivitySinToF::eval(Float time, int idx_tbin) const{
    Float phase = time*omega;
    phase += 2*M_PI*idx_tbin/channels;
    return sin(phase);
}

FloatAD TemporalSensitivitySinToF::eval(FloatAD time, int idx_tbin) const{
    FloatAD phase = time*omega;
    phase += 2*M_PI*idx_tbin/channels;
    return phase.sin();
}

void TemporalSensitivitySinToF::bin_range(Float time, int& idx_start, int& idx_end) const{}
*/
/////////////////////////////////////////////////////
/// CameraTransient
/////////////////////////////////////////////////////
CameraTransient::CameraTransient(int width,
                                 int height,
                                 ptr<float> cam_to_world,
                                 ptr<float> cam_to_ndc,
                                 float clip_near,
                                 int med_id,
                                 ptr<float> velocities,
                                 int duration,
                                 Float tres,
                                 Float tstart,
                                 Float tres_light)
        : duration(duration)
        , tres(tres)
        , tstart(tstart)
        , Camera(width, height, cam_to_world, cam_to_ndc, clip_near, med_id, velocities)
{
//    std::cout<< "# CameraTransient constructor called." << std::endl;
    if (duration > 0){
//        if (tres > 0.0){
//            //this->tend = tstart + tres*duration;
//            if (tres_light < 0.0)
//                tres_light = tres;
//            temp_sens = new TemporalSensitivityTrapezoid(tres, tres_light);
//        }
//        else{
//            temp_sens = new TemporalSensitivitySinToF(duration, -tres);
//        }
        if (tres_light < 0.0)
            tres_light = tres;
        temp_sens = TemporalSensitivity(duration, tres, tres_light);
    }
}

/*CameraTransient::~CameraTransient()
{
    std::cout<< "# CameraTransient destructor called." << std::endl;
    if (valid_transient() && temp_sens){
        std::cout<<temp_sens << std::endl;
        delete temp_sens;
        temp_sens = NULL;
    }
}*/

/*void CameraTransient::bin_range(Float time, int& idx_start, int& idx_end) const{
    assert(valid_transient());
    time -= tstart;
    // Want to satisfy: temp_sens.start_time() < time-tres*idx_tbin < temp_sens.end_time()
    //             <==> (time - end_time) / tres < idx_tbin < (time - start_time) / tres
    idx_start = std::ceil((time - temp_sens.end_time()) / tres);
    idx_end = std::floor((time - temp_sens.start_time()) / tres);
    assert(idx_start <= idx_end);
}*/

int CameraTransient::clip_bin_index(int &idx_start, int &idx_end) const {
    int outtimeSamples = 0;
    if (idx_start < 0){
        outtimeSamples -= idx_start;
        idx_start = 0;
    }
    if (idx_end >= duration){
        outtimeSamples += idx_end - duration + 1;
        idx_end = duration - 1;
    }
    return outtimeSamples;
}