// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolo11-obb torchscript
//      yolo export model=yolo11n-obb.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolo11n-obb.torchscript
// 4. modify yolo11n_obb_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_195 = v_194.view(1, 1, 16384)
//          v_201 = v_200.view(1, 1, 4096)
//          v_207 = v_206.view(1, 1, 1024)
//          v_208 = torch.cat((v_195, v_201, v_207), dim=2)
//          ...
//          v_256 = v_225.view(1, 79, 16384)
//          v_257 = v_240.view(1, 79, 4096)
//          v_258 = v_255.view(1, 79, 1024)
//          v_259 = torch.cat((v_256, v_257, v_258), dim=2)
//          ...
//      after:
//          v_195 = v_194.view(1, 1, -1).transpose(1, 2)
//          v_201 = v_200.view(1, 1, -1).transpose(1, 2)
//          v_207 = v_206.view(1, 1, -1).transpose(1, 2)
//          v_208 = torch.cat((v_195, v_201, v_207), dim=1)
//          ...
//          v_256 = v_225.view(1, 79, -1).transpose(1, 2)
//          v_257 = v_240.view(1, 79, -1).transpose(1, 2)
//          v_258 = v_255.view(1, 79, -1).transpose(1, 2)
//          v_259 = torch.cat((v_256, v_257, v_258), dim=1)
//          return v_259, v_208
//      D. modify area attention for dynamic shape inference
//      before:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, 1024)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, 32, 32)
//          v_107 = v_99.reshape(1, 128, 32, 32)
//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
//          v_109 = (v_106 + v_108)
//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
//      after:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, -1)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, v_95.size(2), v_95.size(3))
//          v_107 = v_99.reshape(1, 128, v_95.size(2), v_95.size(3))
//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
//          v_109 = (v_106 + v_108)
//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
// 5. re-export yolo11-obb torchscript
//      python3 -c 'import yolo11n_obb_pnnx; yolo11n_obb_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolo11n_obb_pnnx.py.pt inputshape=[1,3,1024,1024] inputshape2=[1,3,512,512]
// 7. now you get ncnn model files
//      mv yolo11n_obb_pnnx.py.ncnn.param yolo11n_obb.ncnn.param
//      mv yolo11n_obb_pnnx.py.ncnn.bin yolo11n_obb.ncnn.bin

// YOLO26-obb notes (difference from YOLO11-obb):
//   * The exported ncnn model has ONE output blob "out0" (YOLO11-obb had out0 + out1).
//   * All the post-processing (DFL softmax, anchor offset, distance->box, rotation
//     and score sigmoid) is already baked into the graph, so the C++ side only needs
//     to read the decoded values, threshold them and run NMS.
//   * Static input shape 320x320 -> 2100 anchors (40x40 + 20x20 + 10x10).
//
// the out0 blob is a 2-dim tensor with w=2100 (anchors) h=31
//
//        |  cx  |  cy  |   w  |   h  | class scores (26, already sigmoid) | angle(rad) |
//   row:    0      1      2      3      4 .......................... 29        30
//
//   box(cx,cy,w,h) are in pixel coordinates of the 320x320 letterboxed input.
//   angle is in radians (same value used by cos/sin inside the graph).
//

#include "layer.h"
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <vector>

struct Object
{
    cv::RotatedRect rrect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    std::vector<cv::Point2f> intersection;
    cv::rotatedRectangleIntersection(a.rrect, b.rrect, intersection);
    if (intersection.empty())
        return 0.f;

    return cv::contourArea(intersection);
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rrect.size.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void generate_proposals(const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    // pred: w = num_anchors(2100), h = 4(box) + num_class(scores) + 1(angle)
    const int num_anchors = pred.w;
    const int num_class = pred.h - 5;

    const float* row_cx = pred.row(0);
    const float* row_cy = pred.row(1);
    const float* row_w = pred.row(2);
    const float* row_h = pred.row(3);
    const float* row_angle = pred.row(4 + num_class);

    for (int i = 0; i < num_anchors; i++)
    {
        // find label with max score (scores are already sigmoid-ed inside the graph)
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float s = pred.row(4 + k)[i];
            if (s > score)
            {
                label = k;
                score = s;
            }
        }

        if (score >= prob_threshold)
        {
            // box is already decoded to pixel coordinates (cx, cy, w, h)
            const float cx = row_cx[i];
            const float cy = row_cy[i];
            const float ww = row_w[i];
            const float hh = row_h[i];

            // angle is in radians, cv::RotatedRect wants degrees
            const float angle_rad = row_angle[i];
            const float angle_degree = angle_rad * 180.f / 3.14159265358979323846f;

            Object obj;
            obj.rrect = cv::RotatedRect(cv::Point2f(cx, cy), cv::Size_<float>(ww, hh), angle_degree);
            obj.label = label;
            obj.prob = score;

            objects.push_back(obj);
        }
    }
}

static int detect_yolo26_obb(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net net;

    // the model is a CPU build here, keep vulkan off
    net.opt.use_vulkan_compute = false;

    net.load_param("/home/liran/projects/code/ncnn/models/best_ncnn_model/model.ncnn.param");
    net.load_model("/home/liran/projects/code/ncnn/models/best_ncnn_model/model.ncnn.bin");

    // static input shape of the exported model
    const int target_size = 320;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to a square target_size (model reshapes are static -> must be 320x320)
    float scale = (float)target_size / std::max(img_w, img_h);
    int w = (int)(img_w * scale);
    int h = (int)(img_h * scale);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size square
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Object> proposals;
    generate_proposals(out, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    if (count == 0)
        return 0;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        Object obj = proposals[picked[i]];

        // adjust offset to original unpadded
        obj.rrect.center.x = (obj.rrect.center.x - (wpad / 2)) / scale;
        obj.rrect.center.y = (obj.rrect.center.y - (hpad / 2)) / scale;
        obj.rrect.size.width = (obj.rrect.size.width) / scale;
        obj.rrect.size.height = (obj.rrect.size.height) / scale;

        objects[i] = obj;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "background", "eye_close", "eye_open", "mouth", "nose",
        "ear", "forehead", "front_head", "side_head", "top_head",
        "split_up_milk", "baby_hand", "baby_chin", "baby_chest", "baby_foot",
        "baby_leg", "baby_full_body", "baby_coverd_full_body", "baby_hat", "baby_blinder",
        "baby_arm", "parents_hand", "parents_body", "supine", "prone",
        "lateral"
    };
    const int num_class_names = (int)(sizeof(class_names) / sizeof(class_names[0]));

    static const cv::Scalar colors[] = {
        cv::Scalar(156, 39, 176),
        cv::Scalar(103, 58, 183),
        cv::Scalar(63, 81, 181),
        cv::Scalar(33, 150, 243),
        cv::Scalar(3, 169, 244),
        cv::Scalar(0, 188, 212),
        cv::Scalar(0, 150, 136),
        cv::Scalar(76, 175, 80),
        cv::Scalar(139, 195, 74),
        cv::Scalar(205, 220, 57),
        cv::Scalar(255, 235, 59),
        cv::Scalar(255, 193, 7),
        cv::Scalar(255, 152, 0),
        cv::Scalar(255, 87, 34),
        cv::Scalar(121, 85, 72),
        cv::Scalar(158, 158, 158),
        cv::Scalar(96, 125, 139)
    };
    const int num_colors = (int)(sizeof(colors) / sizeof(colors[0]));

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[obj.label % num_colors];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f  @ %.2f\n", obj.label, obj.prob,
                obj.rrect.center.x, obj.rrect.center.y, obj.rrect.size.width, obj.rrect.size.height, obj.rrect.angle);

        cv::Point2f corners[4];
        obj.rrect.points(corners);
        cv::line(image, corners[0], corners[1], color);
        cv::line(image, corners[1], corners[2], color);
        cv::line(image, corners[2], corners[3], color);
        cv::line(image, corners[3], corners[0], color);
    }

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[obj.label % num_colors];

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label % num_class_names], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rrect.center.x - label_size.width / 2;
        int y = obj.rrect.center.y - label_size.height / 2 - baseLine;
        if (y < 0)
            y = 0;
        if (y + label_size.height > image.rows)
            y = image.rows - label_size.height;
        if (x < 0)
            x = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolo26_obb(m, objects);

    draw_objects(m, objects);

    return 0;
}
