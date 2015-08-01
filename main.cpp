#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/honfree.hpp>

#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <math.h>

#include "cv.h"
#include "highgui.h"
#include "hog.h"

#define SCALE 1.3

std::vector<float> hoge;

std::vector<float> GetHoG(unsigned char* img, int _SIZE_X, int _SIZE_Y, int _CELL_BIN, int _CELL_X, int _CELL_Y, int _BLOCK_X, int _BLOCK_Y){
    
    // HoG：ブロック(3×3セル)分の特徴ベクトルから81次元(9×9)のベクトルVを作る。1セルに何ピクセル(画素)か含まれる。1セル単位で全てのブロックのベクトルを合わせた次元ベクトルを作る。
    int CELL_X = _CELL_X;		// 1セル内の横画素数
    int CELL_Y = _CELL_Y;		// 1セル内の縦画素数
    int CELL_BIN = _CELL_BIN;	// 輝度勾配方向の分割数（普通は９）(20°ずつ)
    int BLOCK_X = _BLOCK_X;		// 1ブロック内の横セル数
    int BLOCK_Y = _BLOCK_Y;		// 1ブロック内の縦セル数
    
    int SIZE_X = _SIZE_X;		// リサイズ後の画像の横幅
    int SIZE_Y = _SIZE_Y;		// リサイズ後の画像の縦幅
    SIZE_X = 40;                // ピクセル数
    SIZE_Y = 40;                // ピクセル数
    
    int CELL_WIDTH = SIZE_X / CELL_X;						// セルの数（横）
    int CELL_HEIGHT = SIZE_Y / CELL_Y;						// セルの数（縦）
    int BLOCK_WIDTH = CELL_WIDTH - BLOCK_X + 1;				// ブロックの数（横）
    int BLOCK_HEIGHT = CELL_HEIGHT - BLOCK_Y + 1;			// ブロックの数（縦）
    
    int BLOCK_DIM = BLOCK_X * BLOCK_Y * CELL_BIN;			// １ブロックの特徴量次元
    int TOTAL_DIM = BLOCK_DIM * BLOCK_WIDTH * BLOCK_HEIGHT;	// HoG全体の次元
    
    double PI = 3.14;
    
    std::vector<float> feat(TOTAL_DIM, 0);
    
    //各セルの輝度勾配ヒストグラム
    std::vector<std::vector<std::vector<double> > > hist;
    hist.resize(CELL_WIDTH);
    for (int i = 0; i < hist.size(); i++){
        hist[i].resize(CELL_HEIGHT);
        for (int j = 0; j < hist[i].size(); j++){
            hist[i][j].resize(CELL_BIN, 0);
        }
    }
    
    //各ピクセルにおける輝度勾配強度mと勾配方向degを算出し、ヒストグラムへ
    //※端っこのピクセルでは、計算しない
    for (int y = 1; y<SIZE_Y - 1; y++){
        for (int x = 1; x<SIZE_X - 1; x++){
            double dx = img[y*SIZE_X + (x + 1)] - img[y*SIZE_X + (x - 1)];
            double dy = img[(y + 1)*SIZE_X + x] - img[(y - 1)*SIZE_X + x];
            double m = sqrt(dx*dx + dy*dy);
            double deg = (atan2(dy, dx) + PI) * 180.0 / PI;	//0.0〜360.0の範囲になるよう変換
            int bin = CELL_BIN * deg / 360.0;
            if (bin < 0) bin = 0;
            if (bin >= CELL_BIN) bin = CELL_BIN - 1;
            hist[(int)(x / CELL_X)][(int)(y / CELL_Y)][bin] += m;
        }
    }
    
    //ブロックごとで正規化
    for (int y = 0; y<BLOCK_HEIGHT; y++){
        for (int x = 0; x<BLOCK_WIDTH; x++){
            
            //このブロックの特徴ベクトル（次元BLOCK_DIM=BLOCK_X*BLOCK_Y*CELL_BIN）
            std::vector<double> vec;
            vec.resize(BLOCK_DIM, 0);
            
            for (int j = 0; j<BLOCK_Y; j++){
                for (int i = 0; i<BLOCK_X; i++){
                    for (int d = 0; d<CELL_BIN; d++){
                        int index = j*(BLOCK_X*CELL_BIN) + i*CELL_BIN + d;
                        vec[index] = hist[x + i][y + j][d];
                    }
                }
            }
            
            //ノルムを計算し、正規化
            double norm = 0.0;
            for (int i = 0; i<BLOCK_DIM; i++){
                norm += vec[i] * vec[i];
            }
            for (int i = 0; i<BLOCK_DIM; i++){
                vec[i] /= sqrt(norm + 1.0);
            }
            
            //featに代入
            for (int i = 0; i<BLOCK_DIM; i++){
                int index = y*BLOCK_WIDTH*BLOCK_DIM + x*BLOCK_DIM + i;
                feat[index] = vec[i];
            }
        }
    }
    return feat;
}

// 顔検出(動画)
int main(int argc, char* argv[]) {
    
    int SIZE_X = 40;    // ピクセル数
    int SIZE_Y = 40;    // ピクセル数
    cv::Mat src, dst, img;
    cv::Mat sample(1, 9*9*((SIZE_X / 5)-2)*((SIZE_Y / 5)-2), CV_32FC1);
    std::vector<float> hog;
    
    CvSVM classifier;
    classifier.load("./svm.xml");
    
    std::string OutputFile = "./image/face.jpg";
    
    // ビデオキャプチャ構造体
    CvCapture *capture = 0;
    // フレーム単位データ
    IplImage *frame = 0;
    // フレーム単位データコピー用
    IplImage *frame_copy = 0;
    // 縦横サイズ
    double height = 240;
    double width = 320;
    // 入力キー，svm結果
    int c, result;
    
    // 正面顔検出器の読み込み
    CvHaarClassifierCascade* cvHCC = (CvHaarClassifierCascade*)cvLoad("/Users/setoyama/Programming/OpenCV/OpenCV-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml");
    
    // 検出に必要なメモリストレージを用意する
    CvMemStorage* cvMStr = cvCreateMemStorage(0);
    
    // 検出情報を受け取るためのシーケンスを用意する
    CvSeq* face;
    
    // 0番号のカメラに対するキャプチャ構造体を生成する
    capture = cvCreateCameraCapture (0);
    
    CvRect* faceRect;
    
    // キャプチャのサイズを設定する。ただし、この設定はキャプチャを行うカメラに依存するので注意する
    cvSetCaptureProperty (capture, CV_CAP_PROP_FRAME_WIDTH, width);
    cvSetCaptureProperty (capture, CV_CAP_PROP_FRAME_HEIGHT, height);
    cvNamedWindow ("capture_face_detect", CV_WINDOW_AUTOSIZE);
    
    // 停止キーが押されるまでカメラキャプチャを続ける
    while (1) {
        frame = cvQueryFrame (capture);
        
        // フレームコピー用イメージ生成
        frame_copy = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, frame->nChannels);
        if(frame->origin == IPL_ORIGIN_TL) {
            cvCopy(frame, frame_copy);
        } else {
            cvFlip(frame, frame_copy);
        }
        
        // 読み込んだ画像のグレースケール化、及びヒストグラムの均一化を行う
        IplImage* gray = cvCreateImage(cvSize(frame_copy->width, frame_copy->height), IPL_DEPTH_8U, 1);
        IplImage* detect_frame = cvCreateImage(cvSize((frame_copy->width / SCALE), (frame_copy->height / SCALE)), IPL_DEPTH_8U, 1);
        cvCvtColor(frame_copy, gray, CV_BGR2GRAY);
        cvResize(gray, detect_frame, CV_INTER_LINEAR);
        cvEqualizeHist(detect_frame, detect_frame);
        
        // 画像中から検出対象の情報を取得する
        face = cvHaarDetectObjects(detect_frame, cvHCC, cvMStr, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30) );
        
        for (int i = 0; i < face->total; i++) {
            
            // 検出情報から顔の位置情報を取得
            faceRect = (CvRect*)cvGetSeqElem(face, i);
            
            // 取得した顔の位置情報に基づき、矩形描画を行う
            cvRectangle(frame_copy,
                        cvPoint(faceRect->x * SCALE, faceRect->y * SCALE),
                        cvPoint((faceRect->x + faceRect->width) * SCALE, (faceRect->y + faceRect->height) * SCALE),
                        CV_RGB(0, 255 ,0),
                        3, CV_AA);
            
            img = cv::cvarrToMat(frame);
            cv::Mat cut_img(img, cv::Rect(faceRect->x * SCALE, faceRect->y * SCALE, faceRect->width * SCALE, faceRect->height * SCALE));
            imwrite(OutputFile, cut_img);
            src = cv::imread("./image/face.jpg", 0);
            cv::resize(src, dst, cv::Size(SIZE_X, SIZE_Y));
            hog = GetHoG(dst.data, dst.cols, dst.rows);
            for(int j = 0; j < hog.size(); j++){
                sample.at<float>(0, j) = hog[j];
            }
            result = classifier.predict(sample);
            if(!result) std::cout << "You are Setoyama" << std::endl;
            if(result) std::cout << "You are not Setoyama!" << std::endl;
        }
        
        // 顔位置に矩形描画を施した画像を表示
        cvShowImage ("capture_face_detect", frame_copy);
        
        // 終了キー入力待ち（タイムアウト付き）
        c = cvWaitKey (10);
        if (c == 'e') {
            break;
        }
    }
    
    // 生成したメモリストレージを解放
    cvReleaseMemStorage(&cvMStr);
    
    // キャプチャの解放
    cvReleaseCapture (&capture);
    
    // ウィンドウの破棄
    cvDestroyWindow("capture_face_detect");
    
    // カスケード識別器の解放
    cvReleaseHaarClassifierCascade(&cvHCC);
    
    return 0;
}