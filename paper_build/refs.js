// Reference database. Entries marked (v) were verified against the publisher / arXiv / proceedings page during this project.
const REFS = {
  iea2025: "International Energy Agency, *Energy and AI*, IEA, Paris, 2025. https://www.iea.org/reports/energy-and-ai (accessed 24 October 2026).",
  masanet2020: "E. Masanet, A. Shehabi, N. Lei, S. Smith, J. Koomey, Recalibrating global data center energy-use estimates, *Science* 367 (6481) (2020) 984–986.",
  radovanovic2023: "A. Radovanović, R. Koningstein, I. Schneider, B. Chen, A. Duarte, B. Roy, D. Xiao, et al., Carbon-aware computing for datacenters, *IEEE Trans. Power Syst.* 38 (2) (2023). arXiv:2106.11750.",
  wiesner2021: "P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen, Let's wait awhile: how temporal workload shifting can reduce carbon emissions in the cloud, in: *Proc. 22nd Int. Middleware Conf. (Middleware '21)*, ACM, 2021. arXiv:2110.13234.",
  sukprasert2024: "T. Sukprasert, A. Souza, N. Bashir, D. Irwin, P. Shenoy, On the limitations of carbon-aware temporal and spatial workload shifting in the cloud, in: *Proc. 19th European Conf. Computer Systems (EuroSys '24)*, ACM, 2024, pp. 924–941. https://doi.org/10.1145/3627703.3650079.",
  acun2023: "B. Acun, B. Lee, F. Kazhamiaka, K. Maeng, U. Gupta, M. Chakkaravarthy, D. Brooks, C.-J. Wu, Carbon Explorer: a holistic framework for designing carbon aware datacenters, in: *Proc. 28th ACM Int. Conf. Architectural Support for Programming Languages and Operating Systems (ASPLOS '23)*, 2023.",
  hanafy2023: "W.A. Hanafy, Q. Liang, N. Bashir, D. Irwin, P. Shenoy, CarbonScaler: leveraging cloud workload elasticity for optimizing carbon-efficiency, *Proc. ACM Meas. Anal. Comput. Syst.* 7 (3) (2023).",
  maji2022: "D. Maji, P. Shenoy, R.K. Sitaraman, CarbonCast: multi-day forecasting of grid carbon intensity, in: *Proc. 9th ACM Int. Conf. Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '22)*, 2022, pp. 198–207. https://doi.org/10.1145/3563357.3564079.",
  maji2022b: "D. Maji, R.K. Sitaraman, P. Shenoy, DACF: day-ahead carbon intensity forecasting of power grids using machine learning, in: *Proc. 13th ACM Int. Conf. Future Energy Systems (e-Energy '22)*, 2022, pp. 188–192. https://doi.org/10.1145/3538637.3538849.",
  yan2025: "L. Yan, L. Wang, S. Liu, Y. Ding, EnsembleCI: ensemble learning for carbon intensity forecasting, arXiv:2505.01959, 2025.",
  li2024: "A. Li, S. Liu, Y. Ding, Uncertainty-aware decarbonization for datacenters, arXiv:2407.02390, 2024.",
  yang2025: "Y. Yang, J. Shi, D. Wang, C. Wu, Z. Han, A conformal prediction-based chance-constrained programming approach for 24/7 carbon-free data center operation scheduling, arXiv:2510.04053, 2025.",
  hu2024: "Q. Hu, Z. Ye, Z. Wang, G. Wang, M. Zhang, Q. Chen, P. Sun, D. Lin, X. Wang, Y. Luo, Y. Wen, T. Zhang, Characterization of large language model development in the datacenter, in: *21st USENIX Symp. Networked Systems Design and Implementation (NSDI 24)*, 2024, pp. 709–729.",
  strubell2019: "E. Strubell, A. Ganesh, A. McCallum, Energy and policy considerations for deep learning in NLP, in: *Proc. 57th Annu. Meeting Assoc. Computational Linguistics*, 2019, pp. 3645–3650.",
  patterson2021: "D. Patterson, J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia, D. Rothchild, D. So, M. Texier, J. Dean, Carbon emissions and large neural network training, arXiv:2104.10350, 2021.",
  schwartz2020: "R. Schwartz, J. Dodge, N.A. Smith, O. Etzioni, Green AI, *Commun. ACM* 63 (12) (2020) 54–63.",
  wu2022: "C.-J. Wu, R. Raghavendra, U. Gupta, B. Acun, N. Ardalani, K. Maeng, et al., Sustainable AI: environmental implications, challenges and opportunities, in: *Proc. Machine Learning and Systems (MLSys)* 4, 2022.",
  luccioni2023: "A.S. Luccioni, S. Viguier, A.-L. Ligozat, Estimating the carbon footprint of BLOOM, a 176B parameter language model, *J. Mach. Learn. Res.* 24 (253) (2023) 1–15.",
  gupta2021: "U. Gupta, Y.G. Kim, S. Lee, J. Tse, H.-H.S. Lee, G.-Y. Wei, D. Brooks, C.-J. Wu, Chasing carbon: the elusive environmental footprint of computing, in: *Proc. IEEE Int. Symp. High-Performance Computer Architecture (HPCA)*, 2021, pp. 854–867.",
  neso: "National Energy System Operator, Carbon Intensity API, https://carbonintensity.org.uk/ (accessed 24 October 2026).",
  powell: "nmpowell, carbon-intensity-forecast-tracking, GitHub repository, https://github.com/nmpowell/carbon-intensity-forecast-tracking (accessed 24 October 2026).",
  acmetrace: "InternLM, AcmeTrace, GitHub repository, https://github.com/InternLM/AcmeTrace (accessed 24 October 2026).",
  hong2016: "T. Hong, P. Pinson, S. Fan, H. Zareipour, A. Troccoli, R.J. Hyndman, Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond, *Int. J. Forecast.* 32 (3) (2016) 896–913.",
  lago2021: "J. Lago, G. Marcjasz, B. De Schutter, R. Weron, Forecasting day-ahead electricity prices: a review of state-of-the-art algorithms, best practices and an open-access benchmark, *Appl. Energy* 293 (2021) 116983.",
  croushore2001: "D. Croushore, T. Stark, A real-time data set for macroeconomists, *J. Econometrics* 105 (1) (2001) 111–130.",
  croushore2011: "D. Croushore, Frontiers of real-time data analysis, *J. Econ. Lit.* 49 (1) (2011) 72–100.",
  nordhaus1987: "W.D. Nordhaus, Forecasting efficiency: concepts and applications, *Rev. Econ. Stat.* 69 (4) (1987) 667–674.",
  mincer1969: "J.A. Mincer, V. Zarnowitz, The evaluation of economic forecasts, in: J.A. Mincer (Ed.), *Economic Forecasts and Expectations*, National Bureau of Economic Research, New York, 1969, pp. 3–46.",
  heath1994: "D.C. Heath, P.L. Jackson, Modeling the evolution of demand forecasts with application to safety stock analysis in production/distribution systems, *IIE Trans.* 26 (3) (1994) 17–30.",
  murphy1993: "A.H. Murphy, What is a good forecast? An essay on the nature of goodness in weather forecasting, *Weather Forecast.* 8 (2) (1993) 281–293.",
  elmachtoub2022: "A.N. Elmachtoub, P. Grigas, Smart \"predict, then optimize\", *Manage. Sci.* 68 (1) (2022) 9–26.",
  mandi2024: "J. Mandi, J. Kotary, S. Berden, M. Mulamba, V. Bucarey, T. Guns, F. Fioretto, Decision-focused learning: foundations, state of the art, benchmark and future opportunities, *J. Artif. Intell. Res.* 80 (2024) 1623–1701.",
  tashman2000: "L.J. Tashman, Out-of-sample tests of forecasting accuracy: an analysis and review, *Int. J. Forecast.* 16 (4) (2000) 437–450.",
  ke2017: "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, LightGBM: a highly efficient gradient boosting decision tree, in: *Adv. Neural Inf. Process. Syst. 30 (NeurIPS 2017)*, 2017, pp. 3146–3154.",
  koenker1978: "R. Koenker, G. Bassett, Regression quantiles, *Econometrica* 46 (1) (1978) 33–50.",
  gneiting2007: "T. Gneiting, A.E. Raftery, Strictly proper scoring rules, prediction, and estimation, *J. Am. Stat. Assoc.* 102 (477) (2007) 359–378.",
  gneiting2014: "T. Gneiting, M. Katzfuss, Probabilistic forecasting, *Annu. Rev. Stat. Appl.* 1 (2014) 125–151.",
  romano2019: "Y. Romano, E. Patterson, E.J. Candès, Conformalized quantile regression, in: *Adv. Neural Inf. Process. Syst. 32 (NeurIPS 2019)*, 2019.",
  vovk2005: "V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*, Springer, New York, 2005.",
  gibbs2021: "I. Gibbs, E. Candès, Adaptive conformal inference under distribution shift, in: *Adv. Neural Inf. Process. Syst. 34 (NeurIPS 2021)*, 2021.",
  kunsch1989: "H.R. Künsch, The jackknife and the bootstrap for general stationary observations, *Ann. Stat.* 17 (3) (1989) 1217–1241.",
  abadi2016: "M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, et al., TensorFlow: a system for large-scale machine learning, in: *12th USENIX Symp. Operating Systems Design and Implementation (OSDI 16)*, 2016, pp. 265–283.",
  kingma2015: "D.P. Kingma, J. Ba, Adam: a method for stochastic optimization, in: *Int. Conf. Learning Representations (ICLR)*, 2015.",
};
function makeCiter() {
  const order = []; const num = (k) => { if (!REFS[k]) throw new Error("unknown ref " + k); let i = order.indexOf(k); if (i < 0) { order.push(k); i = order.length - 1; } return i + 1; };
  const c = (...keys) => {
    const ns = keys.map(num).sort((a, b) => a - b); const parts = []; let i = 0;
    while (i < ns.length) { let j = i; while (j + 1 < ns.length && ns[j + 1] === ns[j] + 1) j++; parts.push(j - i >= 2 ? `${ns[i]}–${ns[j]}` : (j - i === 1 ? `${ns[i]},${ns[j]}` : `${ns[i]}`)); i = j + 1; }
    return "[" + parts.join(",") + "]";
  };
  c.list = () => order.map((k, i) => `[${i + 1}] ${REFS[k]}`);
  c.order = order; return c;
}
module.exports = { REFS, makeCiter };
