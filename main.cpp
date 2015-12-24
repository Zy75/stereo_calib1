#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define HC 5 // horizontal number of corners 
#define VC 6
#define WIDTH 320
#define HEIGHT 240
#define BW 0.031  //  length of one chessboard square
#define WA 11
int main()
{
    VideoCapture capL(0);
    VideoCapture capR(1);

    if(!capL.isOpened())
    {
      return -1;
    }

    if(!capR.isOpened())
    {
      return -1;
    }

    capL.set( CV_CAP_PROP_FRAME_WIDTH, WIDTH );
    capL.set( CV_CAP_PROP_FRAME_HEIGHT, HEIGHT );
    capR.set( CV_CAP_PROP_FRAME_WIDTH, WIDTH );
    capR.set( CV_CAP_PROP_FRAME_HEIGHT, HEIGHT );
 
    namedWindow("cameraL", CV_WINDOW_AUTOSIZE);
    namedWindow("cameraR", CV_WINDOW_AUTOSIZE);

    Mat frameL,frameR;
    bool foundL = false, foundR = false;
    Mat grayL, grayR;
    
    vector<Point3f> obj;
    int i,j;
    for (int j=0; j<HC*VC; j++)
    {
        obj.push_back(Point3f(BW*(float)(j%HC), BW*(float)(j/HC), 0.0f));
    }

    vector<vector<Point2f> > imagePointsL, imagePointsR;
    vector<vector<Point3f> > objectPoints;
    vector<Point2f> cornersL, cornersR;


    for(i=0; i<7; i++) {

      cout << "show chessboard and press q" << endl;
      
      while(1)
      {
          capL >> frameL;
          capR >> frameR;

          imshow("cameraL", frameL);
          imshow("cameraR", frameR);

          int key = waitKey(1);
          if(key == 113)//qボタンが押されたとき
          {
              break;//whileループから抜ける．
          }
      }

      cvtColor(frameL, grayL, CV_BGR2GRAY);
      foundL = findChessboardCorners(grayL, Size(HC,VC), cornersL, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

      cvtColor(frameR, grayR, CV_BGR2GRAY);
      foundR = findChessboardCorners(grayR, Size(HC,VC), cornersR, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

      if(foundL && foundR) {
        imagePointsL.push_back(cornersL);
        imagePointsR.push_back(cornersR);
        objectPoints.push_back(obj);
        cout << "Corners stored" << endl;
      }
      else {
        return -1;
      }

      cornerSubPix(grayL, cornersL, Size(WA, WA), Size(-1, -1), 
                 TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      drawChessboardCorners(grayL, Size(HC,VC), cornersL, foundL);
      

      cornerSubPix(grayR, cornersR, Size(WA, WA), Size(-1, -1), 
                 TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      drawChessboardCorners(grayR, Size(HC,VC), cornersR, foundR);
      
      while(1) {
        imshow("cornersR", grayR);
        imshow("cornersL", grayL);

        int key = waitKey(1);
        if(key == 113)//qボタンが押されたとき
        {
            break;//whileループから抜ける．
        }
      }
    }
   
    Mat CML,CMR,DL,DR;
    Mat R, T, E, F;

    vector<Mat> rvecsL, tvecsL, rvecsR, tvecsR;
    Mat RL, RR, PL, PR, Q;    

    calibrateCamera(objectPoints, imagePointsL, frameL.size(), CML, DL, rvecsL, tvecsL);
    calibrateCamera(objectPoints, imagePointsR, frameR.size(), CMR, DR, rvecsR, tvecsR);

    stereoCalibrate(objectPoints, imagePointsL, imagePointsR, 
                    CML, DL, CMR, DR, frameL.size(), R, T, E, F,
                    cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5), 
                    CV_CALIB_FIX_INTRINSIC);

    stereoRectify(CML, DL, CMR, DR, frameL.size(), R, T, RL, RR, PL, PR, Q);

    FileStorage fs1("mystereocalib.yml", FileStorage::WRITE);
    fs1 << "CML" << CML;
    fs1 << "CMR" << CMR;
    fs1 << "DL" << DL;
    fs1 << "DR" << DR;
    fs1 << "R" << R;
    fs1 << "T" << T;
    fs1 << "RL" << RL;
    fs1 << "RR" << RR;
    fs1 << "PL" << PL;
    fs1 << "PR" << PR;

    fs1.release();

    destroyAllWindows();

    return 0;
}
