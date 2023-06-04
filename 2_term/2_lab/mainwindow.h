#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}
struct Circle
{
    QPoint center;
    int radius;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void paintEvent(QPaintEvent *event);
    void mousePressEvent(QMouseEvent *event);

private:
    Ui::MainWindow *ui;
    QRect rect = QRect(100,100,300,200);
    QVector <QRect> q;
    int rad =55;
    QVector <Circle> v;
    bool good_square_clumb(QRect r);
    bool good_circle_clumb(Circle c);
    bool intersected_circle(Circle c1, Circle c2);
    bool contains(Circle c, QPoint p);
    bool intersected_square_circle(Circle c, QRect r);




};

#endif // MAINWINDOW_H
