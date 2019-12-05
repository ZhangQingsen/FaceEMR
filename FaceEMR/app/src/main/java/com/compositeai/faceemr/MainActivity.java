package com.compositeai.faceemr;

import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.util.Log;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.TextView;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.Button;
import android.content.Intent;
import android.Manifest;
import android.content.pm.PackageManager;
import android.provider.MediaStore;
import android.graphics.Bitmap;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;

import android.os.Environment;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.widget.Toast;

import com.compositeai.predictivemodels.*;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST = 1888;
    private SquareImageView faceImageView;
    private TextView emotionShowView;
    private TextView guessTxtView;
    private static final int MY_CAMERA_PERMISSION_CODE = 100;
    static final int REQUEST_IMAGE_CAPTURE = 1;
    static final int REQUEST_TAKE_PHOTO = 1;
    static final int PIXEL_WIDTH = 48;
    TensorFlowClassifier classifier;
    Button detect;
    Button photoButton;
    static final String make_guess = "Make a Guess";
    static final String treasure_open_path = "assets://treasure_open.png + ";
    static final String treasure_close_path = "assets://treasure_close.png + ";
    static final Bitmap treasure_open = BitmapFactory.decodeFile(treasure_open_path);;
    static final Bitmap treasure_close = BitmapFactory.decodeFile(treasure_close_path);;


    private ImageView treasureImageView;
    private Spinner EMspinner;
    private ArrayAdapter<String> EMSadapter;
    private ArrayList<String> dataList;
    private String guess_EM;
    private String detected_EM;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        detected_EM = make_guess;

        this.faceImageView = (SquareImageView) this.findViewById(R.id.facialImageView);
        photoButton = (Button) this.findViewById(R.id.phototaker);
        photoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dispatchTakePictureIntent();
            }
        });

        detect = (Button) findViewById(R.id.detect);
        detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                detectEmotion();
            }
        });
        Button reset = (Button) findViewById(R.id.reset);
        reset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                clearStatus();
            }
        });
        detect.setEnabled(false);
        this.emotionShowView = (TextView) findViewById(R.id.emotionTxtView);
        this.guessTxtView = (TextView) findViewById(R.id.guessTxtView);


        EMspinner = (Spinner) findViewById(R.id.spinner);

        //为dataList赋值，将下面这些数据添加到数据源中
        dataList = new ArrayList<String>();
        dataList.add(make_guess);
        dataList.add("Angry");
        dataList.add("Disgust");
        dataList.add("Fear");
        dataList.add("Happy");
        dataList.add("Sad");
        dataList.add("Surprise");
        dataList.add("Neutral");
        /*为spinner定义适配器，也就是将数据源存入adapter，这里需要三个参数
        1. 第一个是Context（当前上下文），这里就是this
        2. 第二个是spinner的布局样式，这里用android系统提供的一个样式
        3. 第三个就是spinner的数据源，这里就是dataList*/
        EMSadapter = new ArrayAdapter<String>(this,android.R.layout.simple_spinner_item,dataList);

        //为适配器设置下拉列表下拉时的菜单样式。
        EMSadapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        //为spinner绑定我们定义好的数据适配器
        EMspinner.setAdapter(EMSadapter);

        //为spinner绑定监听器，这里我们使用匿名内部类的方式实现监听器
        EMspinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String settext = ("The emotion you choose is："+EMSadapter.getItem(position));
                Toast.makeText(MainActivity.this,settext,Toast.LENGTH_SHORT).show();
                guess_EM = EMSadapter.getItem(position);
                guessTxtView.setText(guess_EM);
                guessTxtView.setTextColor(getTextColor(guess_EM));
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                guess_EM = make_guess;
                guessTxtView.setText(guess_EM);
                guessTxtView.setTextColor(getTextColor(guess_EM));
            }
        });
        this.treasureImageView = (ImageView) this.findViewById(R.id.treasureImageView);
        //treasureImageView.setImageBitmap(treasure_close);
        treasureImageView.setImageResource(R.drawable.treasure_close);

        loadModel();

    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }


    private void loadModel() {

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier=TensorFlowClassifier.create(getAssets(), "CNN",
                            "opt_em_convnet_5000.pb", "labels.txt", PIXEL_WIDTH,
                            "input", "output_50", true, 7);

                } catch (final Exception e) {
                    //if they aren't found, throw an error!
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }

    /**
     * The main function for emotion detection
     */
    private void detectEmotion(){

        Bitmap image=((BitmapDrawable)faceImageView.getDrawable()).getBitmap();
        Bitmap grayImage = toGrayscale(image);
        Bitmap resizedImage=getResizedBitmap(grayImage,48,48);
        int pixelarray[];

        //Initialize the intArray with the same size as the number of pixels on the image
        pixelarray = new int[resizedImage.getWidth()*resizedImage.getHeight()];

        //copy pixel data from the Bitmap into the 'intArray' array
        resizedImage.getPixels(pixelarray, 0, resizedImage.getWidth(), 0, 0, resizedImage.getWidth(), resizedImage.getHeight());


        float normalized_pixels [] = new float[pixelarray.length];
        for (int i=0; i < pixelarray.length; i++) {
            // 0 for white and 255 for black
            int pix = pixelarray[i];
            int b = pix & 0xff;
            //  normalized_pixels[i] = (float)((0xff - b)/255.0);
            // normalized_pixels[i] = (float)(b/255.0);
            normalized_pixels[i] = (float)(b);

        }
        System.out.println(normalized_pixels);
        Log.d("pixel_values",String.valueOf(normalized_pixels));
        String text=null;

        try{
            final Classification res = classifier.recognize(normalized_pixels);
            //if it can't classify, output a question mark
            if (res.getLabel() == null) {
                text = "Status: "+ "?\n";
            } else {
                //else output its name
                text = String.format("%s: %s\n", "Status ", res.getLabel());
                detected_EM = res.getLabel();
                this.emotionShowView.setTextColor(getTextColor(detected_EM));
                detect.setEnabled(false);
                EMspinner.setEnabled(false);
                photoButton.setEnabled(false);
            }}
        catch (Exception  e){
            System.out.print("Exception:"+e.toString());

        }

        this.faceImageView.setImageBitmap(grayImage);
        this.emotionShowView.setText(text);
        int flag  = checkCorrect();
        String settext = "";
        switch(flag){
            case 1: {
                settext = "Correct!";
                treasureImageView.setImageResource(R.drawable.treasure_open);
            } break;
            case 2: settext = "Try again."; break;
            default: settext = "Try make a guess and check.";
        }
        Toast.makeText(MainActivity.this,settext,Toast.LENGTH_SHORT).show();
    }

    /**
     *
     */
    private void clearStatus(){
        detect.setEnabled(false);
        this.faceImageView.setImageResource(R.drawable.ic_launcher_background);
        this.treasureImageView.setImageResource(R.drawable.treasure_close);
        String text = "Status: ?";
        this.emotionShowView.setText(text);
        this.emotionShowView.setTextColor(getTextColor(text));
        guess_EM = make_guess;
        guessTxtView.setText(guess_EM);
        guessTxtView.setTextColor(getTextColor(guess_EM));
        EMspinner.setEnabled(true);
        EMspinner.setSelection(0);
        photoButton.setEnabled(true);

    }

    /**
     *
     * @param bmpOriginal
     * @return
     */
    // https://stackoverflow.com/questions/3373860/convert-a-bitmap-to-grayscale-in-android?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    //https://stackoverflow.com/questions/15759195/reduce-size-of-bitmap-to-some-specified-pixel-in-android?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    public Bitmap getResizedBitmap(Bitmap image, int bitmapWidth, int bitmapHeight) {
        return Bitmap.createScaledBitmap(image, bitmapWidth, bitmapHeight, true);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            detect.setEnabled(true);
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            faceImageView.setImageBitmap(imageBitmap);
        }
    }

    private int checkCorrect(){
        int result = guess_EM.compareTo(make_guess);
        if (result ==0 )
            result = 0;
        else result = (detected_EM.compareTo(guess_EM) == 0)? 1 : 2;
        // 0 : did not make a guess
        // 1 : make guess correctly
        // 2 : make guess wrong
        return result;
    }

    public static int getTextColor(String text){
        int result = Color.rgb(235, 80, 126);
        if (text.compareTo(make_guess) == 0)
            result = Color.rgb(235, 80, 126);
        else if (text.compareTo("Angry") == 0)
            result = Color.rgb(216, 0, 38);
        else if (text.compareTo("Disgust") == 0)
            result = Color.rgb(225, 145, 24);
        else if (text.compareTo("Fear") == 0)
            result = Color.rgb(51, 30, 30);
        else if (text.compareTo("Happy") == 0)
            result = Color.rgb(255, 125, 0);
        else if (text.compareTo("Sad") == 0)
            result = Color.rgb(44, 52, 92);
        else if (text.compareTo("Surprise") == 0)
            result = Color.rgb(255, 255, 0);
        else if (text.compareTo("Neutral") == 0)
            result = Color.rgb(128, 118, 100);
        else if (text.compareTo("Status: ?") == 0)
            result = Color.rgb(235, 80, 126);
        return result;
    }
}
