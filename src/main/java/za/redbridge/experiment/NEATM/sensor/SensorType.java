package za.redbridge.experiment.NEATM.sensor;

import org.jbox2d.common.MathUtils;

import za.redbridge.experiment.NEATM.sensor.parameter.SensorParameterSpec.BearingSpec;
import za.redbridge.experiment.NEATM.sensor.parameter.SensorParameterSpec.FieldOfViewSpec;
import za.redbridge.experiment.NEATM.sensor.parameter.SensorParameterSpec.OrientationSpec;
import za.redbridge.experiment.NEATM.sensor.parameter.SensorParameterSpec.RangeSpec;
import za.redbridge.experiment.NEATM.sensor.parameter.SensorParameterSpecSet;
import za.redbridge.experiment.NEATM.sensor.parameter.spec.Range;


import static za.redbridge.experiment.NEATM.sensor.parameter.spec.Initializers.copyExistingOrRandom;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.Initializers.random;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.Limiters.clamp;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.Limiters.wrap;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.Ranges.exclusiveRange;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.Ranges.plusMinusHalfPi;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.Ranges.plusMinusPi;
import static za.redbridge.experiment.NEATM.sensor.parameter.spec.Ranges.zeroToPi;

/**
 * Contains the definitions of the valid parameters for each sensor type.
 *
 * Created by jamie on 2014/09/19.
 */
public enum SensorType {

    /**
     * new Spec(initial value, valid range, limiter)
     */

    BOTTOM_PROXIMITY(null),

    PROXIMITY(new SensorParameterSpecSet(
            new BearingSpec(random(), plusMinusPi(), wrap()),
            new OrientationSpec(random(), plusMinusHalfPi(), clamp()),
            new RangeSpec(copyExistingOrRandom(), exclusiveRange(0.01f, 10), clamp()),
            new FieldOfViewSpec(copyExistingOrRandom(), new Range(MathUtils.PI/6, (5*MathUtils.PI)/6, true, true),
                    clamp()))),

    ULTRASONIC(new SensorParameterSpecSet(
            new BearingSpec(random(), plusMinusPi(), wrap()),
            new OrientationSpec(random(), plusMinusHalfPi(), clamp()),
            new RangeSpec(copyExistingOrRandom(), exclusiveRange(0.01f, 25), clamp()),
            new FieldOfViewSpec(copyExistingOrRandom(), new Range(0.1f, MathUtils.PI, true, true),
                    clamp()))),

    LOW_RES_CAM(new SensorParameterSpecSet(
            new BearingSpec(random(), plusMinusPi(), wrap()),
            new OrientationSpec(random(), plusMinusHalfPi(), clamp()),
            new RangeSpec(copyExistingOrRandom(), exclusiveRange(0.01f, 20), clamp()),
            new FieldOfViewSpec(copyExistingOrRandom(), new Range(MathUtils.PI/9, (8*MathUtils.PI)/9, true, true),
                    clamp()))),

    COLOUR_PROXIMITY(new SensorParameterSpecSet(
            new BearingSpec(random(), plusMinusPi(), wrap()),
            new OrientationSpec(random(), plusMinusHalfPi(), clamp()),
            new RangeSpec(copyExistingOrRandom(), exclusiveRange(0.01f, 10), clamp()),
            new FieldOfViewSpec(copyExistingOrRandom(), new Range(MathUtils.PI/6, (5*MathUtils.PI)/6, true, true),
                    clamp())));


    private final SensorParameterSpecSet defaultSpecSet;

    SensorType(SensorParameterSpecSet defaultSpecSet) {
        this.defaultSpecSet = defaultSpecSet;
    }

    public boolean isConfigurable() {
        return defaultSpecSet != null;
    }

    public SensorParameterSpecSet getDefaultSpecSet() {
        return defaultSpecSet;
    }

}
