//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

std::string cococlasses[] = {
    "person",       "bicycle",   "car",           "motorcycle", "airplane",
    "bus",          "train",     "truck",         "boat",       "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",      "bird",
    "cat",          "dog",       "horse",         "sheep",      "cow",
    "elephant",     "bear",      "zebra",         "giraffe"};

void laneyolo::yolo_preprocessing() {
  cv::VideoCapture cap("opencv_resized.mp4");
  if (!cap.isOpened()) {
    std::cout << "Error opening video stream or file" << std::endl;
  }
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
  // cap.set(cv::CAP_PROP_FPS, 100);
  while (1) {
    auto start = std::chrono::steady_clock::now();
    cv::cuda::GpuMat process_frame;
    int i = 0;
    std::vector<torch::Tensor> catTensor;
    std::lock_guard<std::mutex> lk(lock_thread);
    while (i < batchSize) {
      cv::Mat frame;
      cap >> frame;
      if (frame.empty()) break;
      process_frame.upload(frame);
      vecMat.push(frame);
      cv::cuda::GpuMat tensor_sizeimg, tensor_color, tensor_normilize;
      cv::cuda::cvtColor(process_frame, tensor_color, cv::COLOR_BGR2RGB);
      cv::cuda::copyMakeBorder(tensor_color, tensor_sizeimg, paddingt, paddingb,
                               paddingl, paddingr, cv::BORDER_CONSTANT);
      tensor_sizeimg.convertTo(tensor_normilize, CV_32FC3, 1.f / 255.f, 0.f);
      torch::Tensor tensor_img =
          torch::from_blob(tensor_normilize.data,
                           {tensor_normilize.rows, tensor_normilize.cols,
                            tensor_normilize.channels()},
                           torch::Device(torch::kCUDA))
              .permute({2, 0, 1})
              .unsqueeze(0)
              .to(torch::kHalf);
      catTensor.push_back(tensor_img);
      i++;
    }
    if (!catTensor.empty()) {
      // test changes
      // std::reverse(catTensor.begin(), catTensor.end());
      batch_Image = torch::cat({catTensor});
      lock_cond.notify_one();
      auto end = std::chrono::steady_clock::now();
      auto diff = end - start;
      // std::cout << " THE PRE EXECUTION: "
      //<< std::chrono::duration<double, std::milli>(diff).count()
      //<< " ms" << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } else {
      break;
    }
    if (stop_requested()) break;
  }
  emptyFrame = true;
  cap.release();
  std::cout << "Exiting Yolo Preprocess Thread" << std::endl;
  return;
}

void laneyolo::rectcv(const at::TensorAccessor<float, 2>& off_boxes,
                      const at::TensorAccessor<float, 2>& detection_device,
                      std::vector<cv::Rect>& offset_boxvec,
                      std::vector<float>& score_vec) {
  for (int i = 0; i < off_boxes.size(0); i++) {
    offset_boxvec.emplace_back(cv::Rect(
        cv::Point(
            off_boxes[i][0],
            off_boxes[i][1] -
                (paddingt +
                 paddingb)),  // Not the proper way to do mig be good if resized
                              // if image size is greated than 640x640
        cv::Point(
            off_boxes[i][2],
            off_boxes[i][3] -
                (paddingt +
                 paddingb))));  // Not the proper way to do mig be good if
                                // resized if image size is greated than 640x640
    score_vec.emplace_back(detection_device[i][4]);
  }
  return;
}

void laneyolo::yolo_detector() {
  cv::String window_name = "Yolo Display";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  while (1) {
    std::unique_lock<std::mutex> lk(lock_thread);
    lock_cond.wait(lk, [&] { return !vecMat.empty(); });
    auto startdet = std::chrono::steady_clock::now();
    torch::Tensor preds =
        model.forward({batch_Image}).toTuple()->elements()[0].toTensor();
    batch_Image.reset();
    torch::cuda::CUDACachingAllocator::emptyCache();
    std::queue<cv::Mat> tempMat = vecMat;
    std::queue<cv::Mat>().swap(vecMat);
    lk.unlock();
    auto enddet = std::chrono::steady_clock::now();
    auto diffdet = enddet - startdet;
    // std::cout << " THE DETCETION EXECUTION: "
    //<< std::chrono::duration<double, std::milli>(diffdet).count()
    //<< " ms" << std::endl;
    constexpr int item_attr_size = 5;
    int batch_size = preds.size(0);
    // number of classes
    auto num_classes = preds.size(2) - item_attr_size;
    auto conf_mask = preds.select(2, 4).ge(conf_thres).unsqueeze(2);
    auto startpost = std::chrono::steady_clock::now();
    for (size_t i = 0; i < preds.sizes()[0]; ++i) {
      auto detection_device = torch::masked_select(preds[i], conf_mask[i])
                                  .view({-1, num_classes + item_attr_size});
      if (detection_device.size(0) == 0) continue;

      detection_device.slice(1, item_attr_size, item_attr_size + num_classes) *=
          detection_device.select(1, 4).unsqueeze(1);

      torch::Tensor box =
          torch::zeros_like(detection_device.slice(1, 0, 4)).cuda();
      box.select(1, 0) = detection_device.slice(1, 0, 4).select(1, 0) -
                         detection_device.slice(1, 0, 4).select(1, 2).div(2);
      box.select(1, 1) = detection_device.slice(1, 0, 4).select(1, 1) -
                         detection_device.slice(1, 0, 4).select(1, 3).div(2);
      box.select(1, 2) = detection_device.slice(1, 0, 4).select(1, 0) +
                         detection_device.slice(1, 0, 4).select(1, 2).div(2);
      box.select(1, 3) = detection_device.slice(1, 0, 4).select(1, 1) +
                         detection_device.slice(1, 0, 4).select(1, 3).div(2);

      std::tuple<torch::Tensor, torch::Tensor> max_classes =
          torch::max(detection_device.slice(1, item_attr_size,
                                            item_attr_size + num_classes),
                     1);

      // class score
      auto max_conf_score = std::get<0>(max_classes);
      // index
      auto max_conf_index = std::get<1>(max_classes);

      max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
      max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

      // class index(5)
      detection_device =
          torch::cat({box.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);
      // std::cout << detection_device.sizes() << std::endl;
      constexpr int max_wh = 4096;
      auto c = detection_device.slice(1, item_attr_size, item_attr_size + 1) *
               max_wh;
      auto offset_box = detection_device.slice(1, 0, 4) + c;
      std::vector<cv::Rect> offset_boxvec;
      std::vector<float> score_vec;

      auto offset_boxeshost = offset_box.cpu();
      auto detection_host = detection_device.cpu();
      const auto& det_host_array = detection_host.accessor<float, 2>();

      rectcv(offset_boxeshost.accessor<float, 2>(), det_host_array,
             offset_boxvec, score_vec);

      std::vector<int> nmsindices;
      cv::dnn::NMSBoxes(offset_boxvec, score_vec, conf_thres, iou_thres,
                        nmsindices);
      cv::Mat RecMat = tempMat.front();
      // cv::Mat OutFrame;
      // RecMat.copyTo(OutFrame);
      std::vector<float> score_det;
      std::vector<float> class_det;
      for (int index : nmsindices) {
        const auto& b = det_host_array[index];
        cv::rectangle(RecMat,
                      cv::Rect(cv::Point(b[0], b[1]), cv::Point(b[2], b[3])),
                      cv::Scalar(0, 0, 0), 3);
        float confidence = det_host_array[index][4];
        int classes = det_host_array[index][5];
        if (classes <= 23) {
          cv::putText(RecMat, cococlasses[classes], cv::Point(b[0], b[1] - 10),
                      cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 0));
        }
      }
      // cv::addWeighted(RecMat, 0.3, OutFrame, 0.7, 0, OutFrame);
      cv::imshow(window_name, RecMat);
      tempMat.pop();
    }
    if (cv::waitKey(10) >= 0) break;
    auto endpost = std::chrono::steady_clock::now();
    auto diffpost = endpost - startpost;
    // std::cout << " THE POST EXECUTION: "
    //<< std::chrono::duration<double, std::milli>(diffpost).count()
    //<< " ms" << std::endl;
    if (emptyFrame) break;
    if (stop_requested()) break;
  }
  cv::destroyAllWindows();
  std::cout << "Exiting Yolo Postprocess Thread" << std::endl;
  return;
}

void laneyolo::set_batchsize(int batch) {
  batchSize = batch;
  return;
}

void laneyolo::set_iouthres(float iou) {
  iou_thres = iou;
  return;
}

void laneyolo::set_confthres(float conf) {
  conf_thres = conf;
  return;
}

void laneyolo::set_padding(int lpadding, int rpadding, int tpadding,
                           int bpadding) {
  paddingl = lpadding;
  paddingr = rpadding;
  paddingt = tpadding;
  paddingb = bpadding;
  return;
}