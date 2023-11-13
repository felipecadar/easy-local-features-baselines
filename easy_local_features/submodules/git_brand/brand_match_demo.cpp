/* 
   Copyright (C) 2013 Erickson R. Nascimento

   THIS SOURCE CODE IS PROVIDED 'AS-IS', WITHOUT ANY EXPRESS OR IMPLIED
   WARRANTY. IN NO EVENT WILL THE AUTHOR BE HELD LIABLE FOR ANY DAMAGES
   ARISING FROM THE USE OF THIS SOFTWARE.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:


   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   Contact: erickson [at] dcc [dot] ufmg [dot] br

*/


#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/features/integral_image_normal.h>
//#include <pcl/io/ply_io.h>
#include <pcl/surface/organized_fast_mesh.h>

//#include <pcl/visualization/pcl_visualizer.h>
//#include <boost/thread/thread.hpp>

#include "brand.h"

std::stringstream out_timing;
double matching_sum =0, extraction_time =0;

typedef std::vector<std::map<std::string,float> > CSVTable;

//void view_cloud(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals);

std::vector<cv::DMatch> calcAndSaveHammingDistances(std::vector<cv::KeyPoint> kp_query, std::vector<cv::KeyPoint> kp_tgt, 
cv::Mat desc_query, cv::Mat desc_tgt, CSVTable query, CSVTable tgt, std::string file_name)
 {
   //We are going to create a matrix of distances from query to desc and save it to a file 'IMGNAMEREF_IMGNAMETARGET_DESCRIPTORNAME.txt'
   std::vector<cv::DMatch> matches;
   
   std::ofstream oFile(file_name.c_str());
   
   oFile << query.size() << " " << tgt.size() << std::endl;
   
   cv::Mat dist_mat(query.size(),tgt.size(),CV_32S,cv::Scalar(-1));
   
    int c_hits=0;

   for(size_t i=0; i < desc_query.rows; i++)
   {
    int menor = 999, menor_idx=-1, menor_i=-1, menor_j = -1;
     
    for(size_t j = 0; j < desc_tgt.rows; j++)
      {
        int _i = kp_query[i].class_id; //correct idx
        int _j = kp_tgt[j].class_id; //correct idx
        
        //if(_i < 0 || _i >= dist_mat.rows || _j < 0 || _j >= dist_mat.cols)
        //    std::cout << "Estouro: " << _i << " " << _j << std::endl;
        
        if(!(query[_i]["valid"] == 1 && tgt[_i]["valid"] == 1)) //this match does not exist
          continue;

        if(query[_i]["valid"] == 1 && tgt[_j]["valid"] == 1)
        {
            auto start = std::chrono::steady_clock::now();
            dist_mat.at<int>(_i,_j) = cv::norm(desc_query.row(i), desc_tgt.row(j),cv::NORM_HAMMING);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end-start;
            matching_sum+= diff.count();


          if(dist_mat.at<int>(_i,_j) < menor )
          {
                        menor = dist_mat.at<int>(_i,_j);
                        menor_i = _i;
                        menor_j = _j;
                        menor_idx = j;
                    }
        }
        
        //oFile << cv::norm(desc_query.row(i), desc_tgt.row(j),cv::NORM_HAMMING) << " ";
      }

          cv::DMatch d;
          d.distance = menor;
          d.queryIdx = i;
          d.trainIdx = menor_idx;

        if(d.queryIdx >=0 && d.trainIdx >=0)
        {
            matches.push_back(d);
            if(menor_i == menor_j)
                c_hits++;
        }

    }
      
  for(int i=0; i < dist_mat.rows;i++)
    for(int j=0; j < dist_mat.cols; j++)
    {
      oFile << dist_mat.at<int>(i,j) << " ";
    }
    
  oFile << std::endl;   
  oFile.close(); 
    std::cout <<"Correct matches: " << c_hits << " of " << matches.size() << std::endl;
   
   
   return matches;
 }
 


bool load_groundtruth_keypoints_csv(const std::string &filename, std::vector<cv::KeyPoint> &keypoints, CSVTable &csv_data)
{
    std::ifstream fin(filename.c_str());
    int id, valid;
    float x,y;
    std::string line;

    if (fin.is_open())
    {
        std::getline(fin, line); //csv header
        while ( std::getline(fin, line) ) 
        {
            if (!line.empty()) 
            {   
                std::stringstream ss;
                char * pch;

                pch = strtok ((char*)line.c_str()," ,");
                while (pch != NULL)
                {
                    ss << std::string(pch) << " ";
                    pch = strtok (NULL, " ,");
                }

                ss >> id >> x >> y >> valid;

                if(x<0 || y<0)
                    valid = 0;
                                                        
                std::map<std::string,float> csv_line;
                csv_line["id"] = id;
                csv_line["x"] = x;
                csv_line["y"] = y;
                csv_line["valid"] = valid;
                csv_data.push_back(csv_line);
                

                if(valid)
                {
                    cv::KeyPoint kp(cv::Point2f(0,0), 7.0); //6
                    kp.pt.x = x;
                    kp.pt.y = y;
                    kp.class_id = id;
                    //kp.size = keypoint_scale;
                    kp.octave = 0.0;
                    keypoints.push_back(kp);
                }
            }
        }

        fin.close();

    }
    else
    { 
        PCL_WARN("Unable to open ground truth csv file. Gonna use the detector algorithm.\n"); 
        return false;
    }

    return true;
}

bool loadSIFT(const std::string &filename, std::vector<cv::KeyPoint> &keypoints)
{
  keypoints.clear();
  std::ifstream fin(filename.c_str());
  float size, angle, x,y;
  int octave;
  std::string line;
  int i = 0;

  if (fin.is_open())
  {
    std::getline(fin, line); //csv header
    while ( std::getline(fin, line) ) 
    {
      if (!line.empty()) 
      {   
        std::stringstream ss;
        char * pch;

        pch = strtok ((char*)line.c_str()," ,");
        while (pch != NULL)
        {
          ss << std::string(pch) << " ";
          pch = strtok (NULL, " ,");
        }

        ss >> size >> angle >> x >> y >> octave;

        cv::KeyPoint kp(cv::Point2f(x,y), size, angle, 1.0, octave); 
        kp.class_id = i; i++;

        keypoints.push_back(kp);
        
      }
    }

    fin.close();
  }
  else
  { 
    printf("Unable to open keypoints file.\n"); 
    return false;
  }

  printf("[Info] Loaded %d keypoints.\n",keypoints.size());
  return true;
}


void crossCheckMatching( const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                         std::vector<cv::DMatch>& filteredMatches12 )
{
    cv::BFMatcher matcher(cv::NORM_HAMMING,false);
    filteredMatches12.clear();
    std::vector<std::vector<cv::DMatch> > matches12, matches21;
    std::vector<cv::DMatch> matches;
    
    /*matcher.knnMatch( descriptors1, descriptors2, matches12, 1 );
    matcher.knnMatch( descriptors2, descriptors1, matches21, 1 );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            cv::DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                cv::DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
    */
    
    matcher.match(descriptors1,descriptors2, matches);
    filteredMatches12 = matches;


}

void compute_normals(const cv::Mat& cloud, cv::Mat& normals, cv::Mat& img)
{
   pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud( new pcl::PointCloud<pcl::PointXYZ> );
   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcl_cloud_color( new pcl::PointCloud<pcl::PointXYZRGBA> );

   pcl_cloud->clear(); pcl_cloud->clear();
   pcl_cloud->width     = cloud.cols; pcl_cloud_color->width     = cloud.cols;
   pcl_cloud->height    = cloud.rows;  pcl_cloud_color->height    = cloud.rows;
   pcl_cloud->points.resize( pcl_cloud->width * pcl_cloud->height); pcl_cloud_color->points.resize( pcl_cloud->width * pcl_cloud->height);
    
   for(int y = 0; y < cloud.rows; ++y)
   for(int x = 0; x < cloud.cols; ++x)
   {
      pcl_cloud->at(x,y).x = cloud.at<cv::Point3f>(y,x).x;  pcl_cloud_color->at(x,y).x = cloud.at<cv::Point3f>(y,x).x; 
      pcl_cloud->at(x,y).y = cloud.at<cv::Point3f>(y,x).y;  pcl_cloud_color->at(x,y).y = cloud.at<cv::Point3f>(y,x).y;
      pcl_cloud->at(x,y).z = cloud.at<cv::Point3f>(y,x).z;  pcl_cloud_color->at(x,y).z = cloud.at<cv::Point3f>(y,x).z;

      pcl_cloud_color->at(x,y).r = img.at<uchar>(y,x);
      pcl_cloud_color->at(x,y).g = img.at<uchar>(y,x);
      pcl_cloud_color->at(x,y).b = img.at<uchar>(y,x);
      pcl_cloud_color->at(x,y).a = 255;
   }

    /*
    pcl::PolygonMesh mesh;
    pcl::OrganizedFastMesh<pcl::PointXYZ> ofm;

    // Set parameters
    ofm.setInputCloud(pcl_cloud);
    ofm.setMaxEdgeLength(1.5);
    ofm.setTrianglePixelSize(1);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<pcl::PointXYZ>::TRIANGLE_ADAPTIVE_CUT);

    // Reconstruct
    ofm.reconstruct(mesh);
    */
   //pcl::io::savePLYFileASCII("test.ply", *pcl_cloud_color);

   pcl::PointCloud<pcl::Normal>::Ptr pcl_normals (new pcl::PointCloud<pcl::Normal>);
   pcl_normals->clear();
   pcl_normals->width  = pcl_cloud->width;
   pcl_normals->height = pcl_cloud->height;
   pcl_normals->points.resize(pcl_cloud->width * pcl_cloud->height);

   pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
   ne.setInputCloud( pcl_cloud );

   ne.setNormalSmoothingSize( 5 );
   ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
   ne.compute( *pcl_normals );

   normals.create( cloud.size(), CV_32FC3 );

   for(int y = 0; y < pcl_normals->height; ++y)
   for(int x = 0; x < pcl_normals->width; ++x)
   {
      normals.at<cv::Point3f>(y,x).x = pcl_normals->at(x,y).normal_x;
      normals.at<cv::Point3f>(y,x).y = pcl_normals->at(x,y).normal_y; 
      normals.at<cv::Point3f>(y,x).z = pcl_normals->at(x,y).normal_z; 
   }
   
   //view_cloud(pcl_cloud_color, pcl_normals);
}

void create_cloud( const cv::Mat &depth, 
                   float fx, float fy, float cx, float cy, 
                   cv::Mat& cloud )
{
    const float inv_fx = 1.f/fx;
    const float inv_fy = 1.f/fy;

    cloud.create( depth.size(), CV_32FC3 );

    for( int y = 0; y < cloud.rows; y++ )
    {
        cv::Point3f* cloud_ptr = (cv::Point3f*)cloud.ptr(y);
        const uint16_t* depth_prt = (uint16_t*)depth.ptr(y);

        for( int x = 0; x < cloud.cols; x++ )
        {
            float d = (float)depth_prt[x]/1000; // meters
            cloud_ptr[x].x = (x - cx) * d * inv_fx;
            cloud_ptr[x].y = (y - cy) * d * inv_fy;
            cloud_ptr[x].z = d;
        }
    }
}


/*pcl::visualization::PCLVisualizer::Ptr normalsVis (
    pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}
*/

/*void view_cloud(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    pcl::visualization::PCLVisualizer::Ptr viewer = normalsVis(cloud, normals);
    
      while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
    
}
*/

 void save_dist_mat(cv::Mat ref, cv::Mat tgt, int norm, std::string file_name)
 {
    FILE* f = fopen(file_name.c_str(), "w");
    std::cout << "saving " << file_name << std::endl; 

    std::string fmt_str;

    if(norm == cv::NORM_HAMMING)
      fmt_str = "%.0f ";
    else
      fmt_str = "%.8f ";

    fprintf(f, "%d %d\n", ref.rows, tgt.rows);

    for(int i=0; i < ref.rows; i++)
      for(int j=0; j <  tgt.rows; j++)
        fprintf(f, fmt_str.c_str(), cv::norm(ref.row(i), tgt.row(j), norm));
    
    fprintf(f, "\n");

    fclose(f);
 }

int main(int argc, char** argv)
{

   if (argc != 13 )
   {
      std::cerr << "Usage: " << argv[0] << " rgb_image1 depth_image1 rgb_image2 depth_image2 fx fy cx cy.\n";
      return(1);
   }

   // loading images and depth information
   cv::Mat rgb1 = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
   cv::Mat depth1 = cv::imread( argv[2], CV_LOAD_IMAGE_ANYDEPTH );

   cv::Mat rgb2 = cv::imread( argv[3], CV_LOAD_IMAGE_GRAYSCALE );
   cv::Mat depth2 = cv::imread( argv[4], CV_LOAD_IMAGE_ANYDEPTH );

   // intrinsics parameters
   float fx = atof(argv[5]);
   float fy = atof(argv[6]);
   float cx = atof(argv[7]);
   float cy = atof(argv[8]);

   std::string keypoints_ref_file(argv[9]);
   std::string keypoints_tgt_file(argv[10]);

   std::string out1(argv[11]);
   std::string out2(argv[12]);

   std::vector<cv::KeyPoint> keypoints1;
   //detector->detect( rgb1, keypoints1 );
 
   std::vector<cv::KeyPoint> keypoints2;
   //detector->detect( rgb2, keypoints2 );

   CSVTable ref_groundtruth, tgt_groundtruth;

   bool SIFT = keypoints_ref_file.find(".sift") != std::string::npos; //check format .csv or .sift

  if(!SIFT) // groundtruth csv kps
  {
    if (!load_groundtruth_keypoints_csv(keypoints_ref_file, keypoints1, ref_groundtruth))
    {
      std::cout << "Error loading keypoints csv." << std::endl;
      return 1;
    }

    if (!load_groundtruth_keypoints_csv(keypoints_tgt_file, keypoints2, tgt_groundtruth))
    {
      std::cout << "Error loading keypoints csv." << std::endl;
      return 1;
    }
  }
  else
  {
    if (!loadSIFT(keypoints_ref_file, keypoints1))
    {
      std::cout << "Error loading keypoints SIFT." << std::endl;
      return 1;
    }

    if (!loadSIFT(keypoints_tgt_file, keypoints2))
    {
      std::cout << "Error loading keypoints SIFT." << std::endl;
      return 1;
    }    
  }

   // detect keypoint using rgb images
   //cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create( "STAR" );

   // create point cloud and compute normal
   auto start = std::chrono::steady_clock::now();

    std::vector<cv::KeyPoint> kp_ref_copy = keypoints1, kp_tgt_copy = keypoints2;

    int PAD = 100;
    cv::Mat pad_rgb1 = cv::Mat::zeros(rgb1.rows+ 2*PAD, rgb1.cols+ 2*PAD, CV_8UC1);
    cv::Mat pad_rgb2 = cv::Mat::zeros(rgb2.rows+ 2*PAD, rgb2.cols+ 2*PAD, CV_8UC1);

    cv::Mat pad_depth1 = cv::Mat::zeros(rgb1.rows+ 2*PAD, rgb1.cols+ 2*PAD, CV_16U);
    cv::Mat pad_depth2 = cv::Mat::zeros(rgb2.rows+ 2*PAD, rgb2.cols+ 2*PAD, CV_16U);

    rgb1.copyTo(pad_rgb1(cv::Range(PAD, rgb1.rows + PAD), cv::Range(PAD, rgb1.cols + PAD)));
    rgb2.copyTo(pad_rgb2(cv::Range(PAD, rgb2.rows + PAD), cv::Range(PAD, rgb2.cols + PAD)));

    depth1.copyTo(pad_depth1(cv::Range(PAD, rgb1.rows + PAD), cv::Range(PAD, rgb1.cols + PAD)));
    depth2.copyTo(pad_depth2(cv::Range(PAD, rgb2.rows + PAD), cv::Range(PAD, rgb2.cols + PAD)));

    for(int i = 0; i < kp_ref_copy.size(); i++)
    {
      kp_ref_copy[i].pt.x += PAD; kp_ref_copy[i].pt.y += PAD; 
      kp_ref_copy[i].octave = 0;
    }

    for(int i = 0; i < kp_tgt_copy.size(); i++)
    {
      kp_tgt_copy[i].pt.x+= PAD; kp_tgt_copy[i].pt.y+= PAD;
      kp_tgt_copy[i].octave = 0  ;
    }


   cv::Mat cloud1, normals1;
   create_cloud( pad_depth1, fx, fy, cx + PAD, cy + PAD, cloud1 );
   compute_normals( cloud1, normals1, pad_rgb1 );
   
   cv::Mat cloud2, normals2;
   create_cloud( pad_depth2, fx, fy, cx + PAD, cy + PAD, cloud2 );
   compute_normals( cloud2, normals2, pad_rgb2 );

   // extract descriptors
   BrandDescriptorExtractor brand;

   cv::Mat desc1;
   brand.compute( pad_rgb1, cloud1, normals1, kp_ref_copy, desc1 );

   cv::Mat desc2;
   brand.compute( pad_rgb2, cloud2, normals2, kp_tgt_copy, desc2 );

   if(desc1.rows != keypoints1.size() || desc2.rows != keypoints2.size())
  {printf("WARNING, seems the method filtered some keypoints..\n"); getchar(); getchar();}

   auto end = std::chrono::steady_clock::now();
   std::chrono::duration<double> diff = end-start;
   extraction_time+= diff.count();

   // matching descriptors
   std::vector<cv::DMatch> matches; 
   //crossCheckMatching(desc1, desc2, matches);


   //cv::imshow("matches", outimg);
  // cv::waitKey();

  std::stringstream out_distances;
  out_distances << out1 <<"__"<< out2 << "__BRAND";


  if(!SIFT)
  {
    matches = calcAndSaveHammingDistances(keypoints1,keypoints2,desc1,desc2,
    ref_groundtruth,tgt_groundtruth,out_distances.str()+ ".txt");

    cv::Mat outimg;
    cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, outimg, cv::Scalar::all(-1), cv::Scalar::all(-1));
    cv::imwrite(out_distances.str() + ".png", outimg);
    cv::imwrite("matches.png", outimg);
 }
 else
 {
  save_dist_mat(desc1, desc2, cv::NORM_HAMMING, out_distances.str()+ ".dist");
 }

 out_timing << "Time to extract: " << extraction_time <<"s - Time to match: " << matching_sum << "s " << std::endl;

 std::cout << out_timing.str() << std::endl;

   return 0;
}
