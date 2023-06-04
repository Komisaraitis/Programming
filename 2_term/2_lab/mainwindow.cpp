#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMouseEvent>
#include <QPainter>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    QImage image(":/new/prefix1/img/grass.png");
    QBrush brush;
    brush.setTextureImage(image);
    painter.setBrush(brush);
    painter.drawRect(contentsRect());
    painter.drawImage(0,0,image);
    QImage image1(":/new/prefix1/img/water.jfif");
    brush.setTextureImage(image1);
    painter.setBrush(brush);
    painter.drawImage(rect, image1);
    QImage image2(":/new/prefix1/img/klumba.jpg");
    brush.setTextureImage(image2);
    painter.setBrush(brush);
    for (int i = 0; i<q.size(); i++)
        painter.drawImage(q[i], image2);
    for (int i = 0; i<v.size(); i++)
        painter.drawEllipse(v[i].center, v[i].radius, v[i].radius);
}

void MainWindow::mousePressEvent(QMouseEvent *event)
{

    if (event->button() & Qt::RightButton) {
        QRect r(QPoint(0,0), rect.size()*0.2);
        r.moveCenter(event->pos());
        if (good_square_clumb(r))
            q.append(r);

    }
    if (event->button() & Qt::LeftButton) {
       Circle c = {event->pos(), rad};
       if (good_circle_clumb(c))
            v.append(c);

    }



    repaint();
}

bool MainWindow::good_square_clumb(QRect r)
{
    if (rect.contains(r.center()))
        return false;

    for (int i = 0; i<q.size(); i++) {
        if (q[i].intersects(r))
            return false;
    }
    for (int i=0; i<v.size(); i++) {
        if (intersected_square_circle(v[i], r)) {
            if(contains(v[i],r.topLeft()) && contains(v[i],r.topRight()) && contains(v[i],r.bottomLeft()) && contains(v[i],r.bottomRight()))
                return true;
            return false;
        }

    }


    return true;

}

bool MainWindow::good_circle_clumb(Circle c)
{
    if (rect.contains(c.center))
        return false;
    for (int i=0; i<v.size(); i++) {
        if (!intersected_circle(v[i], c))
            return false;
    }
    for (int i=0; i<q.size(); i++) {
        if (intersected_square_circle(c, q[i])) {
            //if(contains(c,q[i].topLeft()) && contains(c,q[i].topRight()) && contains(c,q[i].bottomLeft()) && contains(c,q[i].bottomRight()))
               // return true;
            return false;
        }

    }
    return true;
}

bool MainWindow::intersected_circle(Circle c1, Circle c2)
{
    QPoint c1c2(c2.center-c1.center);
    int dlina = c1c2.x()*c1c2.x() + c1c2.y()*c1c2.y();
    if (dlina <= (c1.radius+c2.radius)*(c1.radius+c2.radius))
        return false;
    return true;

}

bool MainWindow::contains(Circle c, QPoint p)
{
    QPoint tp = c.center - p;
    return(QPoint::dotProduct(tp,tp) <= c.radius * c.radius);
}

bool MainWindow::intersected_square_circle(Circle c, QRect r)
{
    int rc = c.radius;
    QMargins m = QMargins(rc,rc,rc,rc);
    bool corners =
            contains(c,r.topLeft()) ||
            contains(c,r.topRight()) ||
            contains(c,r.bottomLeft()) ||
            contains(c,r.bottomRight());
    return
            corners ||
    r.marginsAdded(m).contains(c.center);


}


