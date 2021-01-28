//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

int getch_nonblock() {
  if (_kbhit()) {
    return _getch();
  } else {
    return -1;
  }
}

int main(int, const char* const[]) {
  laneyolo processorLaneYolo;
  // Uncomment this to test.
  // processorLaneYolo.set_padding(0, 280, 0, 0);
  // processorLaneYolo.set_batchsize(4);
  // processorLaneYolo.set_confthres(0.6);
  // processorLaneYolo.set_iouthres(0.4);
  std::thread th1([&]() { processorLaneYolo.yolo_preprocessing(); });
  std::thread th2([&]() { processorLaneYolo.yolo_detector(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(15000));
  std::thread th3([&]() { processorLaneYolo.lane_detection(); });

  while (1) {
    if (getch_nonblock() == 27) {
      processorLaneYolo.stop();
      break;
    } else {
      // std::cout << processorLaneYolo.get_centerdistance() << std::endl;
      // std::cout << processorLaneYolo.get_leftcurvature() << std::endl;
      // std::cout << processorLaneYolo.get_rightcurvature() << std::endl;
    }
  }
  th1.join();
  th2.join();
  th3.join();
  std::cout << "Thread Joined" << std::endl;
  std::cout << "Exiting Main Function" << std::endl;
  return 0;
}