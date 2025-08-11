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
#include <pcl/io/ply_io.h>
#include <pcl/surface/organized_fast_mesh.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include "brand.h"

std::stringstream out_timing;
double matching_sum = 0, extraction_time = 0;

typedef std::vector<std::map<std::string, float>> CSVTable;

bool load_keypoints(const std::string &filename, std::vector<cv::KeyPoint> &keypoints)
{
    std::ifstream fin(filename.c_str());
    float x,y, size, angle;
    int n = 0;
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

                ss >> x >> y >> size >> angle;

                cv::KeyPoint kp(cv::Point2f(0,0), 7.0); //6
                kp.pt.x = x;
                kp.pt.y = y;
                kp.angle = angle;
                kp.size = size;
                kp.class_id = n;
                kp.octave = 0.0;
                keypoints.push_back(kp);
            }
            n++;
        }

        fin.close();

    }
    else
    { 
        printf("Unable to open ground truth csv file. Gonna use the detector algorithm.\n"); 
        return false;	
    }

    return true;
}

bool write_keypoints(const std::string &filename, const std::vector<cv::KeyPoint> &keypoints)
{
    std::ofstream fout(filename.c_str());
    if (fout.is_open())
    {
        fout << "x,y,size,angle" << std::endl;
        for (size_t i = 0; i < keypoints.size(); ++i)
        {
            fout << keypoints[i].pt.x << "," << keypoints[i].pt.y << "," << keypoints[i].size << "," << keypoints[i].angle << std::endl;
        }
        fout.close();
    }
    else
    {
        return false;
    }
    return true;
}

void compute_normals(const cv::Mat &cloud, cv::Mat &normals, cv::Mat &img)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcl_cloud_color(new pcl::PointCloud<pcl::PointXYZRGBA>);

  pcl_cloud->clear();
  pcl_cloud->clear();
  pcl_cloud->width = cloud.cols;
  pcl_cloud_color->width = cloud.cols;
  pcl_cloud->height = cloud.rows;
  pcl_cloud_color->height = cloud.rows;
  pcl_cloud->points.resize(pcl_cloud->width * pcl_cloud->height);
  pcl_cloud_color->points.resize(pcl_cloud->width * pcl_cloud->height);

  for (int y = 0; y < cloud.rows; ++y)
    for (int x = 0; x < cloud.cols; ++x)
    {
      pcl_cloud->at(x, y).x = cloud.at<cv::Point3f>(y, x).x;
      pcl_cloud_color->at(x, y).x = cloud.at<cv::Point3f>(y, x).x;
      pcl_cloud->at(x, y).y = cloud.at<cv::Point3f>(y, x).y;
      pcl_cloud_color->at(x, y).y = cloud.at<cv::Point3f>(y, x).y;
      pcl_cloud->at(x, y).z = cloud.at<cv::Point3f>(y, x).z;
      pcl_cloud_color->at(x, y).z = cloud.at<cv::Point3f>(y, x).z;

      pcl_cloud_color->at(x, y).r = img.at<uchar>(y, x);
      pcl_cloud_color->at(x, y).g = img.at<uchar>(y, x);
      pcl_cloud_color->at(x, y).b = img.at<uchar>(y, x);
      pcl_cloud_color->at(x, y).a = 255;
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

  pcl::PointCloud<pcl::Normal>::Ptr pcl_normals(new pcl::PointCloud<pcl::Normal>);
  pcl_normals->clear();
  pcl_normals->width = pcl_cloud->width;
  pcl_normals->height = pcl_cloud->height;
  pcl_normals->points.resize(pcl_cloud->width * pcl_cloud->height);

  pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(pcl_cloud);

  ne.setNormalSmoothingSize(5);
  ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
  ne.compute(*pcl_normals);

  normals.create(cloud.size(), CV_32FC3);

  for (int y = 0; y < pcl_normals->height; ++y)
    for (int x = 0; x < pcl_normals->width; ++x)
    {
      normals.at<cv::Point3f>(y, x).x = pcl_normals->at(x, y).normal_x;
      normals.at<cv::Point3f>(y, x).y = pcl_normals->at(x, y).normal_y;
      normals.at<cv::Point3f>(y, x).z = pcl_normals->at(x, y).normal_z;
    }

  //view_cloud(pcl_cloud_color, pcl_normals);
}

void create_cloud(const cv::Mat &depth,
                  float fx, float fy, float cx, float cy,
                  cv::Mat &cloud)
{
  const float inv_fx = 1.f / fx;
  const float inv_fy = 1.f / fy;

  cloud.create(depth.size(), CV_32FC3);

  for (int y = 0; y < cloud.rows; y++)
  {
    cv::Point3f *cloud_ptr = (cv::Point3f *)cloud.ptr(y);
    const uint16_t *depth_prt = (uint16_t *)depth.ptr(y);

    for (int x = 0; x < cloud.cols; x++)
    {
      float d = (float)depth_prt[x] / 1000; // meters
      cloud_ptr[x].x = (x - cx) * d * inv_fx;
      cloud_ptr[x].y = (y - cy) * d * inv_fy;
      cloud_ptr[x].z = d;
    }
  }
}

void print_desc_to_file(cv::Mat desc, std::string filename)
{
  std::ofstream outfile;
  outfile.open(filename);
  outfile << desc.rows << std::endl;
  for (int i = 0; i < desc.rows; i++)
  {
    for (int j = 0; j < desc.cols; j++)
    {
      outfile << unsigned(desc.at<uint8_t>(i, j));
      if (j < desc.cols -1 ){
        outfile << ",";
      }
    }
    outfile << std::endl;
  }
  outfile.close();
}

int main(int argc, char **argv)
{
  // std::cout << "Error loading keypoints csv." << std::endl;
  if (argc < 8)
  {
    std::cerr << "Usage: " << argv[0] << " fx fy cx cy n <rgb_image depth_image keypoints_csv output_file> ... <> \n";
    return (1);
  }

  // intrinsics parameters
  float fx = atof(argv[1]);
  float fy = atof(argv[2]);
  float cx = atof(argv[3]);
  float cy = atof(argv[4]);
  int n = atoi(argv[5]);

  for (int i = 0; i < n; i++)
  { // loading images and depth information
    cv::Mat rgb = cv::imread(argv[(4 * i) + 6], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat depth = cv::imread(argv[(4 * i) + 7], CV_LOAD_IMAGE_ANYDEPTH);
    std::string keypoints_ref_file(argv[(4 * i) + 8]);
    std::string outfilename(argv[(4 * i) + 9]);

    std::vector<cv::KeyPoint> keypoints;

    if (!load_keypoints(keypoints_ref_file, keypoints))
    {
      std::cout << "Error loading keypoints csv." << std::endl;
      return 1;
    }

    cv::Mat cloud, normals;
    create_cloud(depth, fx, fy, cx, cy, cloud);
    compute_normals(cloud, normals, rgb);

    // extract descriptors
    BrandDescriptorExtractor brand;

    cv::Mat desc;
    brand.compute(rgb, cloud, normals, keypoints, desc);
    print_desc_to_file(desc, outfilename);
    write_keypoints(keypoints_ref_file, keypoints);
  }
  return 0;
}
