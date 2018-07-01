package za.redbridge.simulator.khepera;

import za.redbridge.simulator.object.ResourceObject;
import za.redbridge.simulator.object.RobotObject;
import za.redbridge.simulator.object.WallObject;
import za.redbridge.simulator.sensor.AgentSensor;
import za.redbridge.simulator.sensor.sensedobjects.SensedObject;

import java.awt.*;
import java.text.ParseException;
import java.util.List;
import java.util.Map;
/**
 * Return type of whatever is detected (either wall or block) and estimate distance to what has been detected using IR.
 *
 * Differentiates between wall and resourceObject.
 *
 * color = purple
 *
 * Created by Danielle and Alexander on 2018/06/28.
 */

public class ColourProximitySensor extends AgentSensor {

    public static final float RANGE = 0.6f;
    public static final float FIELD_OF_VIEW = 0.4f; // This is a guess

    public static final Color SensorColor = new Color(128, 0, 128, 50);

    public ColourProximitySensor(float bearing) {
        this(bearing, 0.0f, RANGE, FIELD_OF_VIEW);
    }

    public ColourProximitySensor(float bearing, float orientation, float range,
                                 float fieldOfView) {
        super(SensorColor, bearing, orientation, range, fieldOfView);
    }

    @Override
    protected void provideObjectReading(List<SensedObject> objects, List<Double> output) {

        if (!objects.isEmpty()) {
            SensedObject closest = objects.get(0);
            // reading is normalised to be a factor of how close it is
            double reading = Math.min(closest.getDistance() / range, 1.0);

            if(closest.getObject() instanceof ResourceObject){
               output.add(readingCurve(reading)+0.5);
            }
            else if(closest.getObject() instanceof WallObject) {
                output.add(readingCurve(reading));
            }
            else output.add( 0.0);
        }
        else{
            output.add(0.0);
        }
    }

    @Override
    protected Paint getPaint() {
        List<Double> readings = getPreviousReadings();
        if (readings.size() < 3) {
            return super.getPaint();
        }
        return new Color(readings.get(0).floatValue(), readings.get(1).floatValue(),
                readings.get(2).floatValue(), 0.5f);
    }

    //normalises a distance between 0 and 0.5
    protected double readingCurve(double fraction) {
       return fraction/2;
    }
    @Override
    public int getReadingSize() {
        return 1;
    }

    @Override
    public void readAdditionalConfigs(Map<String, Object> map) throws ParseException {
        additionalConfigs = map;
    }



    @Override
    public za.redbridge.simulator.sensor.ColourProximityAgentSensor clone() {

        za.redbridge.simulator.sensor.ColourProximityAgentSensor cloned =
                new za.redbridge.simulator.sensor.ColourProximityAgentSensor(bearing, orientation, range, fieldOfView);

        try {
            cloned.readAdditionalConfigs(additionalConfigs);
        }
        catch (ParseException p) {
            System.out.println("Clone failed.");
            p.printStackTrace();
            System.exit(-1);
        }

        return cloned;
    }

    @Override
    public String toString()
    {
        return "ColourProximitySensor" +
                "\nbearing=" + bearing +
                "\n orientation=" + orientation +
                "\n range=" + range +
                "\n fieldOfView=" + fieldOfView;
    }

    @Override
    public Map<String,Object> getAdditionalConfigs() { return additionalConfigs; }
}


