package com.example.myapplication;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class CsvWriter {

    public static void saveHashMapToCsv(HashMap<Integer, HashMap<Integer, Float>> map, String filePath) {
        try {
//            File outfile = new File(filePath);
//            if (!outfile.exists())
//                if (!outfile.createNewFile())
//                    return;

            FileWriter writer = new FileWriter(filePath);

            for (Map.Entry<Integer, HashMap<Integer, Float>> entry : map.entrySet()) {
                Integer outerKey = entry.getKey();
                HashMap<Integer, Float> innerMap = entry.getValue();

                for (Map.Entry<Integer, Float> innerEntry : innerMap.entrySet()) {
                    Integer innerKey = innerEntry.getKey();
                    Float value = innerEntry.getValue();

                    writer.append(outerKey.toString())
                            .append(",")
                            .append(innerKey.toString())
                            .append(",")
                            .append(value.toString())
                            .append("\n");
                }
            }

//            for (Map.Entry<Integer, HashMap<Integer, Float>> entry : map.entrySet()) {
//                Integer outerKey = entry.getKey();
//                HashMap<Integer, Float> innerMap = entry.getValue();
//                writer.append(outerKey.toString())
//                        .append(",");
//                int cnt = 0;
//                for (Map.Entry<Integer, Float> innerEntry : innerMap.entrySet()) {
//                    Integer innerKey = innerEntry.getKey();
//                    Float value = innerEntry.getValue();
//                    while (cnt < innerKey) {
//                        writer.append(",");
//                        cnt++;
//                    }
//                    writer.append(value.toString());
////                            .append("\n");
//                }
//                writer.append("\n");
//
//            }

            writer.flush();
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}