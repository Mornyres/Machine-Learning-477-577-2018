#define USE_OPENCV_3
//#define USE_OPENCV_2


#ifdef USE_OPENCV_3
   #include <iostream>
   #include <opencv2/highgui.hpp>
   #include <opencv2/imgproc.hpp>
   #include "opencv2/objdetect.hpp"
   #include <opencv2/ml.hpp>
#endif

#ifdef USE_OPENCV_2
   #include <cv.h>
   #include <highgui.h>
   #include <opencv2/ml/ml.hpp>
#endif

#include <iostream>


#ifdef USE_OPENCV_3
using namespace cv::ml;
#endif
using namespace cv;
using namespace std;




string pathName = "digits.png";
int SZ = 20;
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

Mat deskew(Mat& img){
    Moments m = moments(img);
    if(abs(m.mu02) < 1e-2){
        return img.clone();
    }
    float skew = m.mu11/m.mu02;
    Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);

    return imgOut;
} 

void loadTrainTestLabel(string &pathName, vector<Mat> &trainCells, vector<Mat> &testCells,vector<int> &trainLabels, vector<int> &testLabels){

    Mat img = imread(pathName,CV_LOAD_IMAGE_GRAYSCALE);
    int ImgCount = 0;
    for(int i = 0; i < img.rows; i = i + SZ)
    {
        for(int j = 0; j < img.cols; j = j + SZ)
        {
            Mat digitImg = (img.colRange(j,j+SZ).rowRange(i,i+SZ)).clone();
            if(j < int(0.9*img.cols))
            {
                trainCells.push_back(digitImg);
            }
            else
            {
                testCells.push_back(digitImg);
            }
            ImgCount++;
        }
    }
    
    cout << "Image Count : " << ImgCount << endl;
    float digitClassNumber = 0;

    // Loop through array of images, attributing each to a digit for training
    for(int z=0;z<int(0.9*ImgCount);z++){
        if(z % 450 == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
            }
        trainLabels.push_back(digitClassNumber);
    }
    // add digits to testing set
    digitClassNumber = 0;
    for(int z=0;z<int(0.1*ImgCount);z++){
        if(z % 50 == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
            }
        testLabels.push_back(digitClassNumber);
    }
}

void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells,vector<Mat> &deskewedTestCells, vector<Mat> &trainCells, vector<Mat> &testCells){
    

    for(unsigned int i=0;i<trainCells.size();i++){

    	Mat deskewedImg = deskew(trainCells[i]);
    	deskewedTrainCells.push_back(deskewedImg);
    }

    for(unsigned int i=0;i<testCells.size();i++){

    	Mat deskewedImg = deskew(testCells[i]);
    	deskewedTestCells.push_back(deskewedImg);
    }
}

HOGDescriptor hog(
        Size(20,20), //winSize
        Size(10,10), //blocksize
        Size(5,5), //blockStride,
        Size(10,10), //cellSize,
                 9, //nbins,
                  1, //derivAper,
                 -1, //winSigma,
                  0, //histogramNormType,
                0.2, //L2HysThresh,
                  0,//gammal correction,
                  64,//nlevels=64
                  1);
void CreateTrainTestHOG(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainCells, vector<Mat> &deskewedtestCells){

    for(unsigned int y=0;y<deskewedtrainCells.size();y++){
        vector<float> descriptors;
    	hog.compute(deskewedtrainCells[y],descriptors);
    	trainHOG.push_back(descriptors);
    }
   
    for(unsigned int y=0;y<deskewedtestCells.size();y++){
    	
        vector<float> descriptors;
    	hog.compute(deskewedtestCells[y],descriptors);
    	testHOG.push_back(descriptors);
    } 
}
void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
{

    int descriptor_size = trainHOG[0].size();
    
    for(unsigned int i = 0;i<trainHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           trainMat.at<float>(i,j) = trainHOG[i][j]; 
        }
    }
    for(unsigned int i = 0;i<testHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            testMat.at<float>(i,j) = testHOG[i][j]; 
        }
    }
}

void getSVMParams(SVM *svm)
{
    cout << "Kernel type     : " << svm->getKernelType() << endl;
    cout << "Type            : " << svm->getType() << endl;
    cout << "C               : " << svm->getC() << endl;
    cout << "Degree          : " << svm->getDegree() << endl;
    cout << "Nu              : " << svm->getNu() << endl;
    cout << "Gamma           : " << svm->getGamma() << endl;
}

void SVMtrain(Mat &trainMat,vector<int> &trainLabels, Mat &testResponse,Mat &testMat, char method,
    float desiredC,
    float desiredGamma){

      Ptr<SVM> svm = SVM::create();
      svm->setGamma(desiredGamma);
      svm->setC(desiredC);
      
      svm->setKernel(SVM::RBF);
      svm->setType(SVM::C_SVC);
      Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
    
    if (method == 'm'){
      svm->train(td);
    }
    else{
      cout << "Automatic training may take a while... did you mean to specify C and gamma respectively?" << endl;
      svm->trainAuto(td);
    }

    svm->save("model.yml");
    svm->predict(testMat, testResponse);
    getSVMParams(svm);
}

void SVMevaluate(Mat &testResponse,float &count, float &accuracy,vector<int> &testLabels){

    for(int i=0;i<testResponse.rows;i++)
    {
        //cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
        if(testResponse.at<float>(i,0) == testLabels[i]){
            count = count + 1;
        }  
    }
    accuracy = (count/testResponse.rows)*100;
}
int main(int argc, char* argv[]){

    vector<Mat> trainCells;
    vector<Mat> testCells;
    vector<int> trainLabels;
    vector<int> testLabels;
    loadTrainTestLabel(pathName,trainCells,testCells,trainLabels,testLabels);
    	
    vector<Mat> deskewedTrainCells;
    vector<Mat> deskewedTestCells;
    CreateDeskewedTrainTest(deskewedTrainCells,deskewedTestCells,trainCells,testCells);
    
    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > testHOG;
    CreateTrainTestHOG(trainHOG,testHOG,deskewedTrainCells,deskewedTestCells);

    int descriptor_size = trainHOG[0].size();
    cout << "Descriptor Size : " << descriptor_size << endl;
    
    Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
    Mat testMat(testHOG.size(),descriptor_size,CV_32FC1);
  
    ConvertVectortoMatrix(trainHOG,testHOG,trainMat,testMat);
    
    Mat testResponse;

    if (argc == 3){
      // Manual mode
      float desiredC, desiredGamma;
      desiredC = atof(argv[1]);
      desiredGamma = atof(argv[2]);
      SVMtrain(trainMat,trainLabels,testResponse,testMat,'m',desiredC,desiredGamma); 
    }
    
    else if ((argc == 2) && (!strncmp(argv[1], "sweep",5))){
      SVMtrain(trainMat, trainLabels, testResponse, testMat, 'm', 12.5, 0.5625);
    }
    
    else if ((argc==2) && (!strncmp(argv[1],"auto",4))){
    // Auto mode (much slower, more accurate; recursive)
      SVMtrain(trainMat,trainLabels,testResponse,testMat,'a',1,1); 
    }
    
    else{
      cout << "Usage: ./test\t[options]" << endl << "\t\tsweep|auto|(C gamma)"<<endl;
      cerr << "Option not found; exiting..." << endl;
      exit(1);
    }

    float count = 0;
    float accuracy = 0 ;
    SVMevaluate(testResponse,count,accuracy,testLabels);
    
    cout << "Accuracy        : " << accuracy << "%"<< endl;
    return 0;
}
