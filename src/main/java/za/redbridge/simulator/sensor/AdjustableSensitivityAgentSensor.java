package za.redbridge.simulator.sensor;

/**
 * Created by shsu on 2014/09/30.
 */
public abstract class AdjustableSensitivityAgentSensor extends AgentSensor {

    public static final float ENERGY_COST = 7.0f;

    public AdjustableSensitivityAgentSensor() { super(); }

    public AdjustableSensitivityAgentSensor(float bearing, float orientation, float range, float fieldOfView) {
        super(bearing, orientation, range, fieldOfView, ENERGY_COST);
    }

    public abstract void setSensitivity(double sensitivity);

    public abstract double getSensitivity();
}
