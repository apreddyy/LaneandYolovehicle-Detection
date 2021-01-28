//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

float center_point(const float x1, const float x2) { return (x1 + x2) / 2.f; }

float distance(const float x1, const float x2) {
  return ((sqrt(pow(float(x1 - x2), 2.f))));
}

std::vector<float> LinearSpacedArray(const float a, const float b,
                                     const size_t N) {
  float h = (b - a) / static_cast<float>(N - 1);
  std::vector<float> xs(N);
  std::vector<float>::iterator x;
  float val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
    *x = val;
  }
  return xs;
}

void laneprocessing::curvature_sanitycheck(std::vector<float>& polyleft_in,
                                           std::vector<float>& polyright_in,
                                           std::vector<int>& leftx,
                                           std::vector<int>& rightx,
                                           std::vector<int>& main_y) {
  constexpr float xm_per_pix = 3.7f / 350.0f;
  constexpr float ym_per_pix = 30.0f / 360.0f;
  std::vector<float> Plot_ys(360);
  iota(begin(Plot_ys), end(Plot_ys), 0.f);

  std::vector<float> Leftx_out_x;
  std::vector<float> rightx_out_x;
  std::vector<float> Plot_y = LinearSpacedArray(0.f, 20.f, 10);
  Leftx_out_x = polyval_eigen(polyleft_in, Plot_y);
  rightx_out_x = polyval_eigen(polyright_in, Plot_y);
  float Lmean = float(accumulate(Leftx_out_x.begin(), Leftx_out_x.end(), 0.0) /
                      Leftx_out_x.size());
  float Rmean =
      float(accumulate(rightx_out_x.begin(), rightx_out_x.end(), 0.0) /
            rightx_out_x.size());
  float delta_lines = (Rmean - Lmean);

  float L_0 = 2 * polyleft_in[2] * 180 + polyleft_in[1];
  float R_0 = 2 * polyright_in[2] * 180 + polyright_in[1];
  float delta_slope_mid = abs(R_0 - L_0);

  float L_1 = 2 * polyleft_in[2] * 360 + polyleft_in[1];
  float R_1 = 2 * polyright_in[2] * 360 + polyright_in[1];
  float delta_slope_bottom = abs(L_1 - R_1);

  float L_2 = 2 * polyleft_in[2] + polyleft_in[1];
  float R_2 = 2 * polyright_in[2] + polyright_in[1];
  float delta_slope_top = abs(L_2 - R_2);

  std::vector<float> Leftx_sanity;
  std::vector<float> rightx_sanity;

  if (((delta_slope_top <= 0.9) && (delta_slope_bottom <= 0.9) &&
       (delta_slope_mid <= 0.9)) &&
      ((delta_lines > 75))) {
    lastfit::polyleft_last = polyleft_in;
    lastfit::polyright_last = polyright_in;
    Leftx_sanity = polyleft_in;
    rightx_sanity = polyright_in;
  } else {
    Leftx_sanity = lastfit::polyleft_last;
    rightx_sanity = lastfit::polyright_last;
  }

  std::vector<float> Leftx_out;
  std::vector<float> rightx_out;

  Leftx_out = polyval_eigen(Leftx_sanity, Plot_ys);

  rightx_out = polyval_eigen(rightx_sanity, Plot_ys);

  std::vector<float> Leftx_out_m = Leftx_out;
  std::vector<float> rightx_out_m = rightx_out;
  std::vector<float> Plot_ysm = Plot_ys;
  float first_element_L = Leftx_out[359];
  float first_element_R = rightx_out[359];

  float center_x = center_point(first_element_L, first_element_R);
  float center_ix = 320;
  set_centerdistance(distance(center_x, center_ix) * xm_per_pix);

  transform(Leftx_out.begin(), Leftx_out.end(), Leftx_out.begin(),
            bind1st(std::multiplies<float>(), xm_per_pix));

  transform(rightx_out.begin(), rightx_out.end(), rightx_out.begin(),
            bind1st(std::multiplies<float>(), xm_per_pix));

  transform(Plot_ys.begin(), Plot_ys.end(), Plot_ys.begin(),
            bind1st(std::multiplies<float>(), ym_per_pix));

  std::vector<float> left_fit_cr;
  std::vector<float> right_fit_cr;

  left_fit_cr = polyfit_eigen(Plot_ys, Leftx_out, 2);
  set_leftcurvature(float(
      (1 + pow(pow((2 * left_fit_cr[2] * 359 * ym_per_pix + left_fit_cr[1]), 2),
               1.5)) /
      abs(2 * left_fit_cr[2])));

  right_fit_cr = polyfit_eigen(Plot_ys, rightx_out, 2);
  set_rightcurvature(float(
      (1 +
       pow(pow((2 * right_fit_cr[2] * 359 * ym_per_pix + right_fit_cr[1]), 2),
           1.5)) /
      abs(2 * right_fit_cr[2])));

  rightx.insert(rightx.end(), rightx_out_m.begin(), rightx_out_m.end());
  leftx.insert(leftx.end(), Leftx_out_m.begin(), Leftx_out_m.end());
  main_y.insert(main_y.end(), Plot_ysm.begin(), Plot_ysm.end());
  return;
}

// Set Left Curvature.
void laneprocessing::set_leftcurvature(float l) {
  left_curverad = l;
  return;
}
// Set Right Curvature.
void laneprocessing::set_rightcurvature(float r) {
  right_curverad = r;
  return;
}
// Set Vehicle center distance from lanes center to end lanes.
void laneprocessing::set_centerdistance(float c) {
  center_dist = c;
  return;
}
