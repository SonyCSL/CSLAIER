enchant();

// enchant.jsのSurfaceクラスは、HTML5のcanvasと同じ働きをします
// canvasを使用すると画面に直線や円など好きな図形を描くことが出来ます

circluarGraph=function(context,fan,power,memory,temp){
        percent=memory
    
        fan = 360*fan-90
        memory = 360*memory-90
        temp=360*temp-90
        power=360*power-90
        
        //以下、HTML5のcanvasと同じように使えます
        context.beginPath();     //パスを開始
        
        context.clearRect(0,0,240,240)
        

        //(0,50)から(0,200)までの間にグラデーションを設定
        var grad  = context.createRadialGradient(120,120, 10,120,120,100);

        //赤からグラデーション開始
        grad.addColorStop(0,'#020'); 
        grad.addColorStop(0.7,'#020'); 

        //青でグラデーション終了
        grad.addColorStop(1,'rgb(  200,  128,0)'); 
        context.fillStyle = grad; 
                
        radius=100
        start=-90
        end=memory
        x=120
        y=120
        bool = false
        context.moveTo(x,y)
        context.arc(x, y, radius, (start * Math.PI / 180), (end * Math.PI / 180), bool);
        context.fill();
        
        context.closePath();	//パスを終了
    
        context.beginPath();     //パスを開始
        //(0,50)から(0,200)までの間にグラデーションを設定
        var grad  = context.createRadialGradient(120,120, 10,120,120,70);

        grad.addColorStop(0,'#020'); 
        grad.addColorStop(0.7,'#020'); 

        grad.addColorStop(1,'rgb(  200,  0,0)'); 
        context.fillStyle = grad; 
                
        radius=80
        start=-90
        end=power
        x=120
        y=120
        bool = false
        context.moveTo(x,y)
        context.arc(x, y, radius, (start * Math.PI / 180), (end * Math.PI / 180), bool);
        context.fill();
        
        context.closePath();    //パスを終了

        context.beginPath();     //パスを開始
        //(0,50)から(0,200)までの間にグラデーションを設定
        var grad  = context.createRadialGradient(120,120, 10,120,120,50);

        //赤からグラデーション開始
        grad.addColorStop(0,'#020'); 
        grad.addColorStop(0.5,'#020'); 

        grad.addColorStop(1,'rgb(  050,  150,0)'); 
        context.fillStyle = grad; 
                
        radius=60
        start=-90
        end=temp
        x=120
        y=120
        bool = false
        context.moveTo(x,y)
        context.arc(x, y, radius, (start * Math.PI / 180), (end * Math.PI / 180), bool);
        context.fill();
        
        context.closePath();    //パスを終了

        context.beginPath();     //パスを開始
        //(0,50)から(0,200)までの間にグラデーションを設定
        var grad  = context.createRadialGradient(120,120, 10,120,120,50);

        //赤からグラデーション開始
        grad.addColorStop(0,'#020'); 
        grad.addColorStop(0.1,'#020'); 

        //青でグラデーション終了
        grad.addColorStop(1,'rgb(  0,  200,100)'); 
        context.fillStyle = grad; 
                
        radius=40
        start=-90
        end=fan
        x=120
        y=120
        bool = false
        context.moveTo(x,y)
        context.arc(x, y, radius, (start * Math.PI / 180), (end * Math.PI / 180), bool);
        context.fill();
        
        context.closePath();    //パスを終了
        context.fillStyle = '#fa0';
        context.font= 'bold 36px Century Gothic';
        context.fillText( ~~(percent*100)+"%",90,140)
        context.fillStyle = '#fa0';
        context.font= 'bold 10px Century Gothic';
        context.fillText( "Memory ",100,100)


}


var create_gpu_meter = function(fan, power, memory, temp) {
    var game = new Game(240, 240);
    
    game.onload = function(){
        game.rootScene.backgroundColor = "black";
        //Spriteを作ります
        sprite = new Sprite(240,240);        
        //Surfaceを作ります
        surface = new Surface(240,240);
        
        //spriteのimageにsurfaceを代入します
        sprite.image = surface;
        
        //コンテキストを取得します
        context = surface.context;
        a=b=c=d=0;
        sprite.on('enterframe',function(){
            a+= Math.random()*0.01-0.002
            b+= Math.random()*0.01-0.002
            c+= Math.random()*0.01-0.002
            d+= Math.random()*0.01-0.002
            circluarGraph(context,a,b,c,d);
            
        })
        //シーンにサーフェスを追加する
        game.rootScene.addChild(sprite); 
    }
    game.start();
};