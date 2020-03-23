package com.example.irisspeciesclassification;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    private EditText sepalWidth;
    private EditText sepalLength;
    private EditText petalWidth;
    private EditText petalLength;

    private Button getData;

    private TextView showSpecies;

    private Module module;

    final long[] dataShape = {1, 4};
    final String[] species = {"Iris Setosa", "Iris Versicolor", "Iris Virginica"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sepalLength = (EditText) findViewById(R.id.sepalLength);
        sepalWidth = (EditText) findViewById(R.id.sepalWidth);
        petalWidth = (EditText) findViewById(R.id.petalWidth);
        petalLength = (EditText) findViewById(R.id.petalLength);

        getData = (Button) findViewById(R.id.startClassification);

        showSpecies = (TextView) findViewById(R.id.species_view);

        try {
            module = Module.load(assetFilePath(this, "model.pt"));
        } catch (IOException e) {
            Log.e("ASSET_READ_ERROR", "Could not load model. Exception thrown.");
            e.printStackTrace();
            finish();
        }

        getData.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                float sepalW = Float.parseFloat(sepalWidth.getText().toString());
                float sepalL = Float.parseFloat(sepalLength.getText().toString());
                float petalW = Float.parseFloat(petalWidth.getText().toString());
                float petalL = Float.parseFloat(petalLength.getText().toString());

                float floatData[] = {sepalW, sepalL, petalW, petalL};

                Tensor modelInput = Tensor.fromBlob(floatData, dataShape);

                Tensor outputTensor = module.forward(IValue.from(modelInput)).toTensor();
                float[] scores = outputTensor.getDataAsFloatArray();

                int maxScoreIdx = getMaxScoreIdx(scores);

                String speciesName = species[maxScoreIdx];
                float maxScore = scores[maxScoreIdx];

                showSpecies.setVisibility(View.VISIBLE);
                showSpecies.setText(String.format("Predicted class is %s with score of %.4f",
                                    speciesName,
                                    maxScore));
            }
        });
    }

    private int getMaxScoreIdx(float[] scores) {
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }
        return maxScoreIdx;
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
