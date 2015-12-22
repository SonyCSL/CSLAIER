enchant();

var circluarGraph = function(context,fan,power,memory,temp){
        context.clearRect(0,0,228,228);
        var percent=memory;
        fan = 360*fan-90;
        memory = 360*memory-90;
        temp=360*temp-90;
        power=360*power-90;
        
        createCircle(context, 100, 0.7, 'rgb(64,64,0)', 100, 0, 360, true);
        createCircle(context, 100, 0.7, 'rgb(200,128,0)', 100, -90, memory, false);
        createCircle(context, 70, 0.7, 'rgb(200,0,0)', 80, -90, power, false);
        createCircle(context, 50, 0.5, 'rgb(50,150,0)', 60, -90, temp, false);                
        createCircle(context, 50, 0.1, 'rgb(0,200,100)', 40, -90, fan, false);

        context.fillStyle = '#fa0';
        context.font= 'bold 36px Century Gothic';
        context.fillText( ~~(percent*100)+"%",90,140);
        context.fillStyle = '#fa0';
        context.font= 'bold 10px Century Gothic';
        context.fillText( "Memory ",100,100);
};

var createCircle = function(context, r, beginColorOffset, endColorRGB, radius, start, end, bool){
        var x = y = 120;
        context.beginPath();     //パスを開始

        //グラデーションを設定
        var grad  = context.createRadialGradient(120,120,10,120,120,r);

        //グラデーション開始
        grad.addColorStop(0,'#020'); 
        grad.addColorStop(beginColorOffset,'#020'); 

        //グラデーション終了
        grad.addColorStop(1,endColorRGB); 
        context.fillStyle = grad; 

        context.moveTo(x,y);
        context.arc(x, y, radius, (start * Math.PI / 180), (end * Math.PI / 180), bool);
        context.fill();
        context.closePath();    //パスを終了
};

var create_gpu_meter = function(fan,power,power_limit,memory,memory_total,temp) {
    enchant.ENV.USE_TOUCH_TO_START_SCENE = false;
    var game = new Game(228, 228);
    
    var re = /^[\d\.]+/;
    fan = fan.match(re)[0] * 0.001;
    power = power.match(re)[0] / power_limit.match(re)[0];
    memory = memory.match(re)[0] / memory_total.match(re)[0];
    temp = temp.match(re)[0] * 0.001;
    
    game.onload = function(){
        game.rootScene.backgroundColor = "#020";
        //Spriteを作ります
        var sprite = new Sprite(228,228);        
        //Surfaceを作ります
        var surface = new Surface(228,228);
        
        //spriteのimageにsurfaceを代入します
        sprite.image = surface;
        
        //コンテキストを取得します
        var context = surface.context;
        sprite.on('enterframe',function(){
            circluarGraph(context,fan,power,memory,temp);
        })
        //シーンにサーフェスを追加する
        game.rootScene.addChild(sprite); 
    }
    game.start();
}