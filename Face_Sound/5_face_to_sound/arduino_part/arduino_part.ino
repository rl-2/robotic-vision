/* 
 This example shows the arduino is monitoring the signal from Raspberry Pi.
 If it receive a singla, it plays a sound. 
 
 Jieliang (Rodger) Luo
 April 2018
*/

#include <Wire.h>
#include <Zumo32U4.h>

Zumo32U4Buzzer buzzer;

// Store this song in program space using the PROGMEM macro.
// Later we will play it directly from program space, bypassing
// the need to load it all into RAM first.
const char fugue[] PROGMEM =
  "! O5 L16 agafaea dac+adaea fa<aa<bac#a dac#adaea f";
  
boolean play_sound = false;

void setup()       // run once, when the sketch starts
{
  Serial.begin(9600);
}

void loop()        // run over and over again
{
  //Listening from Python
  if(Serial.read() == '1'){
     play_sound = true;
  }
  
  if(play_sound == true){
    // Start playing a tone with frequency 440 Hz at maximum
    // volume (15) for 200 milliseconds.
    buzzer.playFrequency(440, 200, 15);
  
    // Delay to give the tone time to finish.
    delay(1000);
  
    // Start playing note A in octave 4 at maximum volume
    // volume (15) for 2000 milliseconds.
    buzzer.playNote(NOTE_A(4), 2000, 15);
  
    // Wait for 200 ms and stop playing note.
    delay(200);
    buzzer.stopPlaying();
  
    delay(1000);
  
    // Start playing a fugue from program space.
    buzzer.playFromProgramSpace(fugue);
  
    // Wait until it is done playing.
    while(buzzer.isPlaying()){ }
  
    delay(1000);
    
    play_sound = false;
  }
}
