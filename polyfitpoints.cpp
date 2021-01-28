//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include <Eigen/Core>
#include <Eigen/QR>

#include "laneyolo.hpp"

std::vector<float> laneprocessing::polyfit_eigen(const std::vector<float>& xv,
                                                 const std::vector<float>& yv,
                                                 int order) {
  int polifitorder = order + 1;
  Eigen::initParallel();
  Eigen::MatrixXf A = Eigen::MatrixXf::Ones(xv.size(), polifitorder);
  Eigen::VectorXf yv_mapped = Eigen::VectorXf::Map(&yv.front(), yv.size());
  Eigen::VectorXf xv_mapped = Eigen::VectorXf::Map(&xv.front(), xv.size());
  Eigen::VectorXf result;

  assert(xv.size() == yv.size());
  assert(xv.size() >= polifitorder);

  for (int j = 1; j < polifitorder; j++) {
    A.col(j) = A.col(j - 1).cwiseProduct(xv_mapped);
  }

  result = A.householderQr().solve(yv_mapped);
  std::vector<float> coeff;
  coeff.resize(polifitorder);
  for (size_t i = 0; i < order + int(1); i++) coeff[i] = result[i];
  return coeff;
}

std::vector<float> laneprocessing::polyval_eigen(
    const std::vector<float>& oCoeff, const std::vector<float>& oX) {
  int nCount = int(oX.size());
  int nDegree = int(oCoeff.size());
  std::vector<float> oY(nCount);

  for (int i = 0; i < nCount; i++) {
    float nY = 0;
    float nXT = 1;
    float nX = oX[i];
    for (int j = 0; j < nDegree; j++) {
      nY += oCoeff[j] * nXT;
      nXT *= nX;
    }
    oY[i] = nY;
  }

  return oY;
}