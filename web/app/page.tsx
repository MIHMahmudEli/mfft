import Header from "@/components/Header";
import Hero from "@/components/Hero";
import DetectionSection from "@/components/detection/DetectionSection";
import HowItWorks from "@/components/HowItWorks";
import Pricing from "@/components/Pricing";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main>
        <Hero />
        <DetectionSection />
        <HowItWorks />
        <Pricing />
      </main>
      <Footer />
    </div>
  );
}
