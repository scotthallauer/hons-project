package za.redbridge.experiment.HyperNEATM;

import za.redbridge.experiment.NEATM.sensor.SensorModel;
import za.redbridge.experiment.NEATM.sensor.SensorType;

import java.util.ArrayList;
import java.util.List;

/**
 * A builder that stores computes the sensor values for each input node of the substate.
 * Allows the transformation from values to sensor
 * Stores an arrayList for each sensor param
 * @TODO look into location translation to position
 */
public class HyperNEATSensorBuilder {

    /**
     * The ID of this input node.
     */
    private int id;

    /**
     * An arrayList of all connection ranges.
     */
    private List<Float> ranges;

    /**
     * An arrayList of valid weights ie: above minimim.
     */
    private List<Float> weights;
    /**
     * An arrayList of weights.
     */
    private List<Float> FOVs;
    /**
     * An arrayList of orientation.
     */
    private List<Float> orientations;
    /**
     * An arrayList of orientation.
     */
    private List<Float> sensorTypes;

    /**
     * Construct the builder with id.
     * @param id
     */


    public HyperNEATSensorBuilder(int id) {
        this.id =id;
        ranges = new ArrayList<>();
        weights = new ArrayList<>();
        FOVs= new ArrayList<>();
        orientations = new ArrayList<>();
        sensorTypes = new ArrayList<>();
    }

    public void addRanges(Float range) {
        ranges.add(range);
    }

    public void setWeights(Float weight) {
        weights.add(weight);
    }

    public void setFOVs(Float FOV) {
        FOVs.add(FOV);
    }

    public void setOrientations(Float orientation) {
        orientations.add(orientation);
    }
    /**
     * Checks if sensor should be created (ie: has valid connections
     */
    public boolean isValidSensor(){
        return (weights.size()>=1);
    }
    /**
     * creates a sensor model from param arrayLists
     */
    public SensorModel createSensorModel(){

        //Get type of sensor from the connection with max weight
        //@TODO look into if multiple maximums
        Float sensorTypeValue = sensorTypes.get(0);
        Float maxWeight = weights.get(0);
        for(int i =1;i<weights.size();i++){
            if (weights.get(i)>maxWeight){
                maxWeight = weights.get(i);
                sensorTypeValue = sensorTypes.get(i);
            }
        }
        
        return new SensorModel(getSensorType(sensorTypeValue), 0, calculateAverage(orientations),calculateAverage(ranges), calculateAverage(FOVs));
   }
    //maps a float to a sensorType
    //@TODO implement actual map (for now just return
   public SensorType getSensorType(Float type){
        return SensorType.PROXIMITY;
   }
   
   public float calculateAverage(List<Float> arrayList ){
        float sum = 0;
        for(int i =0;i<arrayList.size();i++){
            sum+=arrayList.get(i);
        }
        return sum/arrayList.size();
   }
}
