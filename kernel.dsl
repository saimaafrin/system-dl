void gspmm(graph_t& graph, array2d_t<float>& input, array2d_t<float>& output, bool reverse, bool norm);
void gspmmw_op(graph_t& graph, array2d_t<float>& input, array1d_t<float>& edge_weight, array2d_t<float>& output, op_t op, int64_t reverse);
void gspmmw(graph_t& graph, array1d_t<float>& edge_weight, array1d_t<float>& output, op_t op, int64_t reverse);
void gspmmw2d(graph_t& graph, array2d_t<float>& edge_weight, array2d_t<float>& output, op_t op, int64_t reverse);
void gspmmw_op2d(graph_t& graph, array3d_t<float>& input, array2d_t<float>& edge_weight, array3d_t<float>& output, op_t op, int64_t reverse);
void gsddmm(graph_t& graph, array1d_t<float>& input_left, array1d_t<float>& input_right, array1d_t<float>& output, op_t op, int64_t reverse);
void gsddmm2d(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array2d_t<float>& output, op_t op, int64_t reverse);
void gsddmme(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op, int64_t reverse);
void gsddmme2d(graph_t& graph, array3d_t<float>& input_left, array3d_t<float>& input_right, array2d_t<float>& output, op_t op, int64_t reverse);
void sddmme_model(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op);
void spmmw_model(graph_t& graph, array2d_t<float>& input, array1d_t<float>& edge_weight, array1d_t<float>& bias_array, array2d_t<float>& output, op_t op, int64_t reverse);
void spmmw_model_without_bias(graph_t& graph, array2d_t<float>& input, array1d_t<float>& edge_weight, array2d_t<float>& output, op_t op, int64_t reverse);