package za.redbridge.simulator.factories;

import org.jbox2d.collision.AABB;
import org.jbox2d.common.Vec2;
import org.jbox2d.dynamics.World;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import za.redbridge.simulator.PlacementArea;
import za.redbridge.simulator.config.Config;
import za.redbridge.simulator.object.ResourceObject;


import java.io.FileReader;
import java.io.FileWriter;
import java.util.Map;

/**
 * Created by shsu on 2014/08/29.
 */
public class ConfigurableResourceFactory extends Config implements ResourceFactory {

    private static final int DEFAULT_LARGE_QUANTITY = 10;
    private static final int DEFAULT_MEDIUM_QUANTITY = 15;
    private static final int DEFAULT_SMALL_QUANTITY = 15;

    private static final float DEFAULT_SMALL_WIDTH = 0.4f;
    private static final float DEFAULT_SMALL_HEIGHT = 0.4f;
    private static final float DEFAULT_SMALL_MASS = 1.0f;
    private static final int DEFAULT_SMALL_PUSHING_BOTS = 1;
    private static final double DEFAULT_SMALL_VALUE = 1;

    private static final float DEFAULT_MEDIUM_WIDTH = 0.6f;
    private static final float DEFAULT_MEDIUM_HEIGHT = 0.6f;
    private static final float DEFAULT_MEDIUM_MASS = 3.0f;
    private static final int DEFAULT_MEDIUM_PUSHING_BOTS = 2;
    private static final double DEFAULT_MEDIUM_VALUE = 2;

    private static final float DEFAULT_LARGE_WIDTH = 0.8f;
    private static final float DEFAULT_LARGE_HEIGHT = 0.8f;
    private static final float DEFAULT_LARGE_MASS = 5.0f;
    private static final int DEFAULT_LARGE_PUSHING_BOTS = 3;
    private static final double DEFAULT_LARGE_VALUE = 3;

    private ResourceSpec smallResourceSpec;
    private ResourceSpec mediumResourceSpec;
    private ResourceSpec largeResourceSpec;

    private int NumPlacements =0;

    public ConfigurableResourceFactory() {
        smallResourceSpec = new ResourceSpec(DEFAULT_SMALL_QUANTITY, DEFAULT_SMALL_WIDTH,
                DEFAULT_SMALL_HEIGHT, DEFAULT_SMALL_MASS, DEFAULT_SMALL_PUSHING_BOTS,
                DEFAULT_SMALL_VALUE);
        
        mediumResourceSpec = new ResourceSpec(DEFAULT_MEDIUM_QUANTITY, DEFAULT_MEDIUM_WIDTH,
                DEFAULT_MEDIUM_HEIGHT, DEFAULT_MEDIUM_MASS, DEFAULT_MEDIUM_PUSHING_BOTS,
                DEFAULT_MEDIUM_VALUE);
        
        largeResourceSpec = new ResourceSpec(DEFAULT_LARGE_QUANTITY, DEFAULT_LARGE_WIDTH,
                DEFAULT_LARGE_HEIGHT, DEFAULT_LARGE_MASS, DEFAULT_LARGE_PUSHING_BOTS,
                DEFAULT_LARGE_VALUE);
    }

    @Override
    public void placeInstances(PlacementArea.ForType<ResourceObject> placementArea, World world) {
     //   String file = "configPlacementNumber"+Random.number(0,max);
        String makeNewFile =  "../configPlacementNumber"+NumPlacements;
        NumPlacements++;
        JSONObject json = new JSONObject();
        JSONArray small = placeInstances(smallResourceSpec, placementArea, world);
        json.put("small",small);
        JSONArray med = placeInstances(mediumResourceSpec, placementArea, world);
        json.put("med",med);
        JSONArray large = placeInstances(largeResourceSpec, placementArea, world);
        json.put("large",large);
        try {
            FileWriter file = new FileWriter(makeNewFile+".json");
            //File Writer creates a file in write mode at the given location
            file.write(json.toJSONString());

            //write function is use to write in file,
            //here we write the Json object in the file
            file.flush();

        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }
    public void placeInstancesFile(PlacementArea.ForType<ResourceObject> placementArea, World world) {
        String file = "configPlacementNumber10.json";
        JSONParser parser = new JSONParser();
        JSONObject jsonObject;
        try {
            Object obj = parser.parse(new FileReader(file));
            jsonObject = (JSONObject) obj;
            placeInstanceReadFile(smallResourceSpec,placementArea,world,(JSONArray) jsonObject.get("small"));
            placeInstanceReadFile(mediumResourceSpec,placementArea,world,(JSONArray) jsonObject.get("med"));
            placeInstanceReadFile(largeResourceSpec,placementArea,world,(JSONArray) jsonObject.get("large"));
        }

        catch(Exception e){
                e.printStackTrace();
        }

        //parsing the JSON string inside the file that we created earlier.






    }
    public void placeInstanceReadFile(ResourceSpec spec, PlacementArea.ForType<ResourceObject> placementArea, World world,JSONArray json){
        for (int i = 0; i < spec.quantity; i++) {
            JSONObject object = (JSONObject)json.get(i);
            Vec2 lower = new Vec2( ((Double) object.get("LowerBoundX")).floatValue(), ((Double) object.get("LowerBoundY")).floatValue());
            Vec2 upper = new Vec2( ((Double) object.get("UpperBoundX")).floatValue(), ((Double) object.get("UpperBoundY")).floatValue());
            PlacementArea.Space space = new PlacementArea.Space(new AABB(lower,upper),((Double) object.get("angle")).floatValue());

            ResourceObject resource = new ResourceObject(world, space.getPosition(),
                    space.getAngle(), spec.width, spec.height, spec.mass, spec.pushingBots,
                    spec.value);


            placementArea.placeObject(space, resource);
        }
    }

    private JSONArray placeInstances(ResourceSpec spec,
            PlacementArea.ForType<ResourceObject> placementArea, World world) {
        JSONArray array = new JSONArray();
        for (int i = 0; i < spec.quantity; i++) {
            PlacementArea.Space space =
                    placementArea.getRandomRectangularSpace(spec.width, spec.height);

            JSONObject spaceJSON = new JSONObject();
            spaceJSON.put("angle",space.getAngle());
            spaceJSON.put("LowerBoundX",space.getAabb().lowerBound.x);
            spaceJSON.put("LowerBoundY",space.getAabb().lowerBound.y);
            spaceJSON.put("UpperBoundX",space.getAabb().upperBound.x);
            spaceJSON.put("UpperBoundY",space.getAabb().upperBound.y);
            array.add(spaceJSON);

            ResourceObject resource = new ResourceObject(world, space.getPosition(),
                    space.getAngle(), spec.width, spec.height, spec.mass, spec.pushingBots,
                    spec.value);


            placementArea.placeObject(space, resource);
        }
        return array;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void configure(Map<String, Object> resourceConfigs) {
        Map<String, Object> smallConfig = (Map<String, Object>) resourceConfigs.get("small");
        Map<String, Object> mediumConfig = (Map<String, Object>) resourceConfigs.get("medium");
        Map<String, Object> largeConfig = (Map<String, Object>) resourceConfigs.get("large");

        smallResourceSpec = readConfig(smallConfig, "resourceConfig:small:", DEFAULT_SMALL_VALUE);
        mediumResourceSpec =
                readConfig(mediumConfig, "resourceConfig:medium:", DEFAULT_MEDIUM_VALUE);
        largeResourceSpec = readConfig(largeConfig, "resourceConfig:large:", DEFAULT_LARGE_VALUE);
    }
    
    private ResourceSpec readConfig(Map<String, Object> config, String path, double value) {
        int quantity = 0;
        int pushingBots = 0;
        float width = 0;
        float height = 0;
        float mass = 0;
        
        Number quantityField = (Number) config.get("quantity");
        if (checkFieldPresent(quantityField, path + "quantity")) {
            quantity = quantityField.intValue();
        }
        
        Number widthField = (Number) config.get("width");
        if (checkFieldPresent(widthField, path + "width")) {
            width = widthField.floatValue();
        }
        
        Number heightField = (Number) config.get("height");
        if (checkFieldPresent(heightField, path + "height")) {
            height = heightField.floatValue();
        }
        
        Number massField = (Number) config.get("mass");
        if (checkFieldPresent(heightField, path + "mass")) {
            mass = massField.floatValue();
        }
        
        Number pushingBotsField = (Number) config.get("pushingBots");
        if (checkFieldPresent(pushingBotsField, path + "pushingBots")) {
            pushingBots = pushingBotsField.intValue();
        }
        
        return new ResourceSpec(quantity, width, height, mass, pushingBots, value);
    }

    @Override
    public int getNumberOfResources() {
        return smallResourceSpec.quantity + mediumResourceSpec.quantity
                + largeResourceSpec.quantity;
    }

    @Override
    public double getTotalResourceValue(){
        return smallResourceSpec.getTotalValue() + mediumResourceSpec.getTotalValue() +
                largeResourceSpec.getTotalValue();
    }

    private static class ResourceSpec {
        private final int quantity;
        private final float width;
        private final float height;
        private final float mass;
        private final int pushingBots;
        private final double value;

        ResourceSpec(int quantity, float width, float height, float mass, int pushingBots,
                double value) {
            this.quantity = quantity;
            this.width = width;
            this.height = height;
            this.mass = mass;
            this.pushingBots = pushingBots;
            this.value = value;
        }
        
        double getTotalValue() {
            return quantity * value;
        }
    }
}
