#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include "omp.h"
#include "opencv2/core/internal.hpp"





#define PI 3.14159265
using namespace cv;
using namespace std;
/*
static void convert_neon8px_int(const cv::Mat& input, cv::Mat& output)
{
    uint8_t __restrict * dest = output.data;
    uint8_t __restrict * src  = input.data;
    int numPixels             = 307200;

    uint8x8_t rfac = vdup_n_u8 (77);
    uint8x8_t gfac = vdup_n_u8 (151);
    uint8x8_t bfac = vdup_n_u8 (28);

    uint16x8_t  temp;
    uint8x8_t result;
    uint8x8x3_t rgb;
register int n = numPixels /8 ;


    for (register int i = 0; i < n; i++, src += 8*3, dest += 8)
    {
            
        rgb  = vld3_u8 (src);

        temp = vmull_u8 (rgb.val[0],      bfac);
        temp = vmlal_u8 (temp,rgb.val[1], gfac);
        temp = vmlal_u8 (temp,rgb.val[2], rfac);

        result = vshrn_n_u16 (temp, 8);
        vst1_u8 (dest, result);
 
    }
}
class convert_neon_asm_8px : public cv::ParallelLoopBody
{
	private:
      const cv::Mat& in;
      cv::Mat& out;
      
public :
    convert_neon_asm_8px(const cv::Mat& input, cv::Mat& output): in(input), out(output)
    {	
    }

    void operator() (const cv::Range& range) const
    {
        uint8_t __restrict * dest = out.data + range.start;
        uint8_t __restrict * src  = in.data  + range.start * 3;

        int numPixels             = 307200;

        __asm__ volatile("lsr          %2, %2, #3      \n"
                     "# build the three constants: \n"
                     "mov         r4, #28          \n" // Blue channel multiplier
                     "mov         r5, #151         \n" // Green channel multiplier
                     "mov         r6, #77          \n" // Red channel multiplier
                     "vdup.8      d4, r4           \n"
                     "vdup.8      d5, r5           \n"
                     "vdup.8      d6, r6           \n"
                     "0:                           \n"
                     "# load 8 pixels:             \n"
                     "vld3.8      {d0-d2}, [%1]!   \n"
                     "# do the weight average:     \n"
                     "vmull.u8    q7, d0, d4       \n"
                     "vmlal.u8    q7, d1, d5       \n"
                     "vmlal.u8    q7, d2, d6       \n"
                     "# shift and store:           \n"
                     "vshrn.u16   d7, q7, #8       \n" // Divide q3 by 256 and store in the d7
                     "vst1.8      {d7}, [%0]!      \n"
                     "subs        %2, %2, #1       \n" // Decrement iteration count
                     "bne         0b            \n" // Repeat until iteration count is not zero
                     :
                     : "r"(dest), "r"(src), "r"(numPixels)
                     : "r4", "r5", "r6"
                     );
    }
};


static void convert_neon_asm_8px(const cv::Mat& input, cv::Mat& output)
{
    uint8_t __restrict * dest = output.data;
    uint8_t __restrict * src  = input.data;
    int numPixels             = 307200;

    __asm__ volatile("lsr          %2, %2, #3      \n"
                     "# build the three constants: \n"
                     "mov         r4, #28          \n" // Blue channel multiplier
                     "mov         r5, #151         \n" // Green channel multiplier
                     "mov         r6, #77          \n" // Red channel multiplier
                     "vdup.8      d4, r4           \n"
                     "vdup.8      d5, r5           \n"
                     "vdup.8      d6, r6           \n"
                     "0:                           \n"
                     "# load 8 pixels:             \n"
                     "vld3.8      {d0-d2}, [%1]!   \n"
                     "# do the weight average:     \n"
                     "vmull.u8    q7, d0, d4       \n"
                     "vmlal.u8    q7, d1, d5       \n"
                     "vmlal.u8    q7, d2, d6       \n"
                     "# shift and store:           \n"
                     "vshrn.u16   d7, q7, #8       \n" // Divide q3 by 256 and store in the d7
                     "vst1.8      {d7}, [%0]!      \n"
                     "subs        %2, %2, #1       \n" // Decrement iteration count
                     "bne         0b            \n" // Repeat until iteration count is not zero
                     :
                     : "r"(dest), "r"(src), "r"(numPixels)
                     : "r4", "r5", "r6"
                     );
}*/



int main(int argc, char **argv)  {




    double temps_initial, temps_final, temps_cpu;
    Mat img_color;
    Mat eyesROI;
    
    CascadeClassifier cascade_face;
    CascadeClassifier cascade_eyes;
    CascadeClassifier cascade_eyes_glaces;
    
    const float scale_factor(1.2f);
    const int min_neighbors(4);
    vector<Rect> faces;
    int elapsed_milliseconds, elapsed_milliseconds1, elapsed_milliseconds2;
    int counter=0;
    int somme_time;
    Point yeux[2];
    double start, end, t2;
    
    

    if (!cascade_face.load("lbpcascade_frontalface.xml") ) {
        cout << "Couldn't load face_cascade" << endl;
        exit(-1);
    }

    if (!cascade_eyes_glaces.load("haarcascade_eye_tree_eyeglasses.xml") ) {
        cout << "Couldn't load eyes_glasses_cascade" << endl;
        exit(-1);
    }
       
                    
      
      img_color = imread("BioID_0001.pgm", 1); 
 
    
      //assert(img_color.empty() == false);

      Mat img(img_color.size(), CV_8UC1);
start=(double)cvGetTickCount();
        //convert_neon_asm_8px(img_color, img);
      cvtColor(img_color,img, CV_BGR2GRAY);
end=(double)cvGetTickCount();
        t2=(end - start)/((double)cvGetTickFrequency()*1000);
	printf("the overall time processing =%g ms\n", t2); 
                
      equalizeHist(img, img);
      


      cascade_face.detectMultiScale(img, faces, 1.2f, 4 , 0 ,Size(70,70));
                

          
           
             for (int n = 0; n < faces.size(); n++)
                               { 
					Point center_face(faces[n].x + faces[n].width*0.5, faces[n].y + faces[n].height*0.5 );
                                       rectangle(img_color, faces[n], Scalar(255,0,0), 8);
                                        
				        eyesROI=img(faces[n]);
					vector<Rect> eyes;
temps_initial = clock ();

                                        cascade_eyes_glaces.detectMultiScale(eyesROI, eyes, 1.2f, 4, 0 |CV_HAAR_SCALE_IMAGE,Size(25,25));
 
       

					
						//#pragma omp parallel for num_threads(num_threads)
                                            for( int j = 0; j < eyes.size(); j++ )
					    {
						Point center(faces[n].x + eyes[j].x + eyes[j].width*0.5, faces[n].y + eyes[j].y + eyes[j].height*0.5 );
                                          	int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
             		                        circle(img_color, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
                                                yeux[j]=center;
 					    }

					   // line(img_color, yeux[0], yeux[1], Scalar(0,255,0), 3, CV_AA);
					
	                                


/*face orientation */


        double diff_x = abs(yeux[0].x - yeux[1].x); 
        double diff_y = abs( yeux[0].y  -   yeux[1].y);
        double theta = abs(atan(diff_x / diff_y) * 180 / PI);
        //printf ("Theta is %f degrees\n", theta);
        double orientation;
        double milieu = (yeux[0].x + yeux[1].x)/2;
        if (milieu < center_face.x)  {
         orientation =  - ( 90 -  theta);
         printf ("The left orientation of face is %f degrees\n", orientation);
         }
        else 
            {
              orientation =  90 -  theta; 
              printf ("The right orientation of face is %f degrees\n", orientation);
            }



 }

        imshow("Face Detector", img_color);
        bool bOk=imwrite("test10.jpg",img_color);
            
                                                 // Wait for a keystroke in the window
   

        waitKey(0);

 
    return 0;
}
