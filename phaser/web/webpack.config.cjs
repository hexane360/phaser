const path = require('path');
//const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  mode: "development",
  devtool: 'eval-source-map',
  entry: {
    manager: './src/manager.tsx',
    dashboard: './src/dashboard.tsx',
    //"manager-widget": './src/manager-widget.tsx',
    //"dashboard-widget": './src/dashboard-widget.tsx',
  },
  output: {
    filename: "[name].js",
    path: path.resolve(__dirname, "static", "dist"),
  },
  module: {
    rules: [
        {
          test: /\.tsx?$/i,
          use: 'ts-loader',
          include: [path.resolve(__dirname, 'src')],
        },
        {
          test: /\.css$/i,
          use: [
            "style-loader",
            {
              loader: "css-loader",
              options: {
                importLoaders: 1,
                modules: {
                  namedExport: false,
                  exportLocalsConvention: "as-is",
                  auto: /\.module\.css$/i,
                  localIdentName: '[name]__[local]',
                },
              }
            },
            "postcss-loader",
          ]
        },
        {
          test: /\.svg$/i,
          type: 'asset/resource',
          resourceQuery: /url/, // *.svg?url
        },
        {
          test: /\.svg$/i,
          issuer: /\.[jt]sx?$/i,
          use: [{
            loader: "@svgr/webpack",
            options: { typescript: true },
          }],
          resourceQuery: { not: [/url/] },
        },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js', '.tsx', '.jsx'],
  },
  plugins: [],
};
