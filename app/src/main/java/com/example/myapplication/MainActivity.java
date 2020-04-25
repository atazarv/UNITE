package com.example.myapplication;
import com.opencsv.CSVReader;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;



public class MainActivity extends AppCompatActivity {
    TextView textView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    /*    List<data1>  data1s = new ArrayList<>();
        try {
            //csv file containing data
            //String strFile = "R.raw.datafile.csv";
            //CSVReader reader = new CSVReader(new FileReader(strFile));
            CSVReader reader = new CSVReader(new InputStreamReader(getResources().openRawResource(R.raw.data_201909121857)));
            String [] nextLine;
            int lineNumber = 0;
            //reader.readLine();
            while ((nextLine = reader.readNext()) != null) {
                lineNumber++;
                if (lineNumber==1) continue;

                System.out.println("Line # " + lineNumber);

                // nextLine[] is an array of values from the line
                String [] tokens = nextLine[0].split("\t");
                System.out.println("PPG " + tokens[1] + " etc...");

                data1 sample = new data1();
                sample.setTimestamp(Double.parseDouble(tokens[0]));
                sample.setPpg(Double.parseDouble(tokens[1]));
                data1s.add(sample);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        */
        //readsourcefile();
        
/*
        String fileName = "data_201909121857.csv";
        File file = new File(fileName);
        try {
            Scanner inputStream = new Scanner(file);
            while (inputStream.hasNext()){
                String data = inputStream.next();
                System.out.println(data);
            }
            inputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
*/

        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        Python py = Python.getInstance();
        PyObject pyf = py.getModule("Feature_Extraction");
        PyObject object = pyf.callAttr("main");

        textView = findViewById(R.id.sina);

        System.out.println("****************************");
        System.out.println(object);
        textView.setText(object.toString());

    }
}
