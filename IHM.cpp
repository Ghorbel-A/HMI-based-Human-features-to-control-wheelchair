#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#define PI 3.14159265
cv::Mat roiImg;


using namespace cv;
using namespace std;

Mat img1; Mat img2;
Mat templ; Mat result;
const char* image_window = "Source Image";
const char* result_window = "Result window";

Mat template_img1 = imread("open.jpg", CV_LOAD_IMAGE_COLOR);
Mat template_img2 = imread("closed.jpg", CV_LOAD_IMAGE_COLOR);

int match_method=0;
int eye_open=0;
int eye_close=0;

Point matchLoc;


void MatchingMethod(Mat img_display, Mat template_img, int id )
{
  /// Source image to display
  
  int match_method = CV_TM_CCORR_NORMED;

  Mat result_mat;

  matchTemplate(img_display, template_img, result_mat, match_method);
  normalize(result_mat, result_mat, 0, 1, NORM_MINMAX, -1, cv::Mat());
  double minVal; double maxVal; 
  Point minLoc, maxLoc;
  minMaxLoc(result_mat, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

  
  //Justing checkin the match template value reaching the threashold
  if(id == 0 && (minVal < 0))
	{
	eye_open=eye_open+1;
	if(eye_open == 10)
		{
		std::cout<<"Eye Open"<<std::endl;
		eye_open=0;
		eye_close=0;
		}
	}
   else if(id == 1 && (minVal < 0))
	eye_close=eye_close+1;
	if(eye_close == 10)
		{
		std::cout<<"Eye Closed"<<std::endl;
		eye_close=0;
		
		}

		


  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )  matchLoc = minLoc;
  else matchLoc = maxLoc;
  

  
  return;
}

void detect_blink(cv::Mat roi)
{


	try
	{  	
	MatchingMethod(roi, template_img1,0);
 	MatchingMethod(roi, template_img2,1);

	}

	catch( cv::Exception& e )

	{
		std::cout<<"An exception occued"<<std::endl;
	}
}



int main(int argc, char **argv) {
	
    CvCapture* capture;
    Mat img;

    Mat eyesROI, result_mat;
    FILE* fichier = NULL;
    CascadeClassifier cascade_face;
    CascadeClassifier cascade_eyes;
    CascadeClassifier cascade_eyes_glaces;
    
    const float scale_factor(1.2f);
    const int min_neighbors(4);
    vector<Rect> faces;
    
    int elapsed_milliseconds;
    fichier= fopen("temps.txt", "w");
    int counter=0;
    int somme_time=0;
    Point yeux[2];
    double temps_cvcolor, start, end,t1, t2;


    
    
    
    if (!cascade_face.load("lbpcascade_frontalface.xml") ) {
        cout << "Couldn't load face_cascade" << endl;
        exit(-1);
    }
    if (!cascade_eyes.load("eyes.xml") ) {
        cout << "Couldn't load eyes_cascade" << endl;
        exit(-1);
    }
    if (!cascade_eyes_glaces.load("haarcascade_eye_tree_eyeglasses.xml") ) {
        cout << "Couldn't load eyes_glasses_cascade" << endl;
        exit(-1);
    }


    
	capture = cvCaptureFromCAM(0);


        while(true)        
		{ 
          		   
          Mat img_color = cvQueryFrame(capture);
                    
		    {
			if(!img_color.empty())
			{
				counter++;
                                
                                    
				/*debut de mesure du temps d'execution*/
				/*start= (double)cvGetTickCount();  */
				cvtColor(img_color,img, CV_BGR2GRAY);
                /* end= (double)cvGetTickCount();
				temps_cvcolor= (end-start)/((double)cvGetTickFrequency()*1000);// microsecondes
				printf("\n\n The overall time processing=%g ms\n", temps_cvcolor);*/

                equalizeHist(img, img);
				cascade_face.detectMultiScale(img, faces, 1.2f, 4 , 0 ,Size(70,70));
		
				for (int n = 0; n < faces.size(); n++)
                                { 
					Rect r = faces[n];
					Point center_face(faces[n].x + faces[n].width*0.5, faces[n].y + faces[n].height*0.5 );				
       				rectangle(img_color, faces[n], Scalar(255,0,0), 8);
					Rect rect = Rect(r.x,r.y + (r.height/5.5),r.width,r.height/3.0);
					Mat roiImg = img_color(rect);

				
					detect_blink(img_color);

				    eyesROI=img(faces[n]);

					vector<Rect> eyes;
                    cascade_eyes_glaces.detectMultiScale(eyesROI, eyes, 1.2f, 4, 0 |CV_HAAR_SCALE_IMAGE,Size(25,25));
                                        
                                            for( int j = 0; j < eyes.size(); j++ )
					    {
						Point center(faces[n].x + eyes[j].x + eyes[j].width*0.5, faces[n].y + eyes[j].y + eyes[j].height*0.5 );
                                          	int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
             		                        circle(img_color, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
                                                yeux[j]=center;
 					    }
					    line(img_color, yeux[0], yeux[1], Scalar(0,255,0), 3, CV_AA);

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
				
		           
		        }

		        else
		        { 
				printf(" --(!) No captured frame -- Break!"); 
				break; 
		        }
        
                    
		    imshow("VJ Face Detector", img_color);
 		    if (fichier != NULL)  // si l'ouverture a réussi
		    {
				fprintf(fichier, "elapsed time: %d ms\n", elapsed_milliseconds);
					
	            }
		    else 
			        printf("Impossible d'ouvrir le fichier temps.txt");

		    int c = waitKey(10);
		    if( (char)c == 'c' ) 
				        break; 
		          
	                }


             }
             

    
    return 0;
}


